// Unit tests for the 8 LLM training improvements:
// - Box-Muller gaussian RNG
// - all_finite_full SIMD NaN/Inf scan
// - Tiled GEMM (gemm_rowmajor_ABt_bias)
// - Dropout kernels
// - Vectorized activations (GELU, SiLU forward/backward)
// - Final LayerNorm in training + inference
// - Dropout integration (training with dropout > 0)
// - Checkpoint persistence round-trip for final LN tensors

#include "transformer-improvements-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/transformer_kernels.h"
#include "../../../Backend/Machine Learning/Networks/transformer_ops.h"
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

// In-memory token-id DataInput for test networks (same as nn-test.cpp).
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

	virtual bool hasTokenIdInput() const { return true; }
	virtual bool hasTokenIdExpectedOutput() const { return true; }

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

// Helper: create a small token-LM transformer decoder network with given config tweaks.
struct SmallTransformerSetup
{
	glades::NNInfo* info;
	glades::NNetwork* net;
	TestTokenIdInput* di;

	SmallTransformerSetup(unsigned int vocab, unsigned int dModel, unsigned int nHeads,
	                      unsigned int dFF, unsigned int seed, int nLayers = 1)
	    : info(NULL), net(NULL), di(NULL)
	{
		const unsigned int padTokenId = vocab - 1u;
		std::vector<unsigned int> toks;
		for (unsigned int i = 0; i < 8u; ++i)
			toks.push_back(i % (vocab - 1u));

		di = new TestTokenIdInput();
		di->setTrainTokens(toks, static_cast<int>(padTokenId));
		di->mirrorTrainToTest();

		glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		for (int l = 0; l < nLayers; ++l)
			hidden.push_back(new glades::HiddenLayerInfo(static_cast<int>(dModel), 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);

		info = new glades::NNInfo("ut_transformer_improvements", in, hidden, out);
		net = new glades::NNetwork(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net->setSeed(seed);
		net->getTerminatorMutable().setEpoch(3);
		net->getTerminatorMutable().setAccuracy(0);

		glades::TrainingConfig& cfg = net->getTrainingConfigMutable();
		cfg.transformer.enableTokenEmbedding = true;
		cfg.transformer.vocabSizeOverride = static_cast<int>(vocab);
		cfg.transformer.tieEmbeddings = true;
		cfg.transformer.padTokenId = static_cast<int>(padTokenId);
		cfg.transformer.nHeadsOverride = static_cast<int>(nHeads);
		cfg.transformer.nKVHeadsOverride = static_cast<int>(nHeads);
		cfg.transformer.dFFOverride = static_cast<int>(dFF);
		cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
		cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
		cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
		cfg.transformer.ropeTheta = 10000.0f;
		cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
	}

	~SmallTransformerSetup()
	{
		delete net;
		delete info;
		delete di;
	}
};

} // anonymous namespace

void TransformerImprovementsUnitTest()
{
	printf("============================================================\n");
	printf("Transformer Improvements Test Suite\n");
	printf("============================================================\n");

	// ===== 1. Box-Muller Gaussian RNG =====
	printf("-----------------------------------\n");
	printf("Box-Muller Gaussian RNG\n");
	printf("-----------------------------------\n");
	{
		glades::rng::Engine eng;
		glades::rng::seed_engine(eng, 12345u);

		const int N = 10000;
		double sum = 0.0;
		double sumSq = 0.0;
		for (int i = 0; i < N; ++i)
		{
			const float v = glades::rng::standard_normal(eng);
			G_assert(__FILE__, __LINE__, "==============Gaussian: non-finite sample==============", std::isfinite(v));
			sum += static_cast<double>(v);
			sumSq += static_cast<double>(v) * static_cast<double>(v);
		}

		const double mean = sum / N;
		const double var = sumSq / N - mean * mean;
		// Mean should be near 0 (within 0.05 for N=10000)
		G_assert(__FILE__, __LINE__, "==============Gaussian: mean not near 0==============", fabs(mean) < 0.05);
		// Variance should be near 1 (within 0.1 for N=10000)
		G_assert(__FILE__, __LINE__, "==============Gaussian: variance not near 1==============", fabs(var - 1.0) < 0.1);

		// Test normal(mean, stddev) wrapper
		glades::rng::seed_engine(eng, 99u);
		double sum2 = 0.0;
		for (int i = 0; i < N; ++i)
			sum2 += static_cast<double>(glades::rng::normal(eng, 5.0f, 0.02f));
		const double mean2 = sum2 / N;
		G_assert(__FILE__, __LINE__, "==============Gaussian: normal(5,0.02) mean not near 5==============", fabs(mean2 - 5.0) < 0.01);
	}

	// ===== 2. all_finite_full kernel =====
	printf("-----------------------------------\n");
	printf("all_finite_full SIMD NaN/Inf scan\n");
	printf("-----------------------------------\n");
	{
		// All-finite array
		std::vector<float> good(100, 1.0f);
		G_assert(__FILE__, __LINE__, "==============all_finite_full: good array returns false==============",
		         glades::transformer_kernels::all_finite_full(&good[0], good.size()));

		// Empty array is finite
		G_assert(__FILE__, __LINE__, "==============all_finite_full: empty returns false==============",
		         glades::transformer_kernels::all_finite_full(NULL, 0));

		// Single NaN
		good[50] = std::numeric_limits<float>::quiet_NaN();
		G_assert(__FILE__, __LINE__, "==============all_finite_full: NaN not detected==============",
		         !glades::transformer_kernels::all_finite_full(&good[0], good.size()));

		// Restore and add +Inf
		good[50] = 1.0f;
		good[99] = std::numeric_limits<float>::infinity();
		G_assert(__FILE__, __LINE__, "==============all_finite_full: +Inf not detected==============",
		         !glades::transformer_kernels::all_finite_full(&good[0], good.size()));

		// -Inf
		good[99] = -std::numeric_limits<float>::infinity();
		G_assert(__FILE__, __LINE__, "==============all_finite_full: -Inf not detected==============",
		         !glades::transformer_kernels::all_finite_full(&good[0], good.size()));

		// NaN in scalar tail (test array of size 3 — no AVX loop, only scalar tail)
		float tail[3] = {1.0f, 2.0f, 3.0f};
		G_assert(__FILE__, __LINE__, "==============all_finite_full: small array good fails==============",
		         glades::transformer_kernels::all_finite_full(tail, 3));
		tail[1] = std::numeric_limits<float>::quiet_NaN();
		G_assert(__FILE__, __LINE__, "==============all_finite_full: small NaN not detected==============",
		         !glades::transformer_kernels::all_finite_full(tail, 3));

		// Large array (exercises multiple AVX iterations)
		std::vector<float> large(1024, 3.14f);
		G_assert(__FILE__, __LINE__, "==============all_finite_full: large clean array fails==============",
		         glades::transformer_kernels::all_finite_full(&large[0], large.size()));
		large[1023] = std::numeric_limits<float>::quiet_NaN();
		G_assert(__FILE__, __LINE__, "==============all_finite_full: large NaN at end not detected==============",
		         !glades::transformer_kernels::all_finite_full(&large[0], large.size()));
	}

	// ===== 3. Tiled GEMM correctness =====
	printf("-----------------------------------\n");
	printf("gemm_rowmajor_ABt_bias tiled GEMM\n");
	printf("-----------------------------------\n");
	{
		// Y = X * W^T + bias
		// X: [T=3, inSize=4], W: [outSize=2, inSize=4], bias: [outSize=2]
		const unsigned int T = 3;
		const unsigned int inSz = 4;
		const unsigned int outSz = 2;
		const float X[T * inSz] = {
			1.0f, 2.0f, 3.0f, 4.0f,
			5.0f, 6.0f, 7.0f, 8.0f,
			0.5f, 0.5f, 0.5f, 0.5f
		};
		const float W[outSz * inSz] = {
			1.0f, 0.0f, 1.0f, 0.0f,   // row 0: dot with X gives x[0]+x[2]
			0.0f, 1.0f, 0.0f, 1.0f    // row 1: dot with X gives x[1]+x[3]
		};
		const float bias[outSz] = {0.1f, 0.2f};
		float Y[T * outSz];
		memset(Y, 0, sizeof(Y));

		glades::transformer_kernels::gemm_rowmajor_ABt_bias(X, T, inSz, W, bias, outSz, outSz, Y);

		// Row 0: [1+3+0.1, 2+4+0.2] = [4.1, 6.2]
		G_assert(__FILE__, __LINE__, "==============GEMM Y[0,0] wrong==============", fabs(Y[0] - 4.1f) < 1e-5f);
		G_assert(__FILE__, __LINE__, "==============GEMM Y[0,1] wrong==============", fabs(Y[1] - 6.2f) < 1e-5f);
		// Row 1: [5+7+0.1, 6+8+0.2] = [12.1, 14.2]
		G_assert(__FILE__, __LINE__, "==============GEMM Y[1,0] wrong==============", fabs(Y[2] - 12.1f) < 1e-5f);
		G_assert(__FILE__, __LINE__, "==============GEMM Y[1,1] wrong==============", fabs(Y[3] - 14.2f) < 1e-5f);
		// Row 2: [0.5+0.5+0.1, 0.5+0.5+0.2] = [1.1, 1.2]
		G_assert(__FILE__, __LINE__, "==============GEMM Y[2,0] wrong==============", fabs(Y[4] - 1.1f) < 1e-5f);
		G_assert(__FILE__, __LINE__, "==============GEMM Y[2,1] wrong==============", fabs(Y[5] - 1.2f) < 1e-5f);

		// No bias
		float Y2[T * outSz];
		memset(Y2, 0, sizeof(Y2));
		glades::transformer_kernels::gemm_rowmajor_ABt_bias(X, T, inSz, W, NULL, 0, outSz, Y2);
		G_assert(__FILE__, __LINE__, "==============GEMM no-bias Y[0,0] wrong==============", fabs(Y2[0] - 4.0f) < 1e-5f);
		G_assert(__FILE__, __LINE__, "==============GEMM no-bias Y[0,1] wrong==============", fabs(Y2[1] - 6.0f) < 1e-5f);

		// Larger test to exercise tiling (T=16, inSize=128, outSize=16)
		{
			const unsigned int bigT = 16;
			const unsigned int bigK = 128;
			const unsigned int bigO = 16;
			std::vector<float> bigX(bigT * bigK, 0.0f);
			std::vector<float> bigW(bigO * bigK, 0.0f);
			std::vector<float> bigBias(bigO, 0.0f);
			std::vector<float> bigY(bigT * bigO, 0.0f);
			std::vector<float> refY(bigT * bigO, 0.0f);

			// Fill with deterministic pattern (cast to int to avoid unsigned underflow)
			for (size_t i = 0; i < bigX.size(); ++i)
				bigX[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.1f;
			for (size_t i = 0; i < bigW.size(); ++i)
				bigW[i] = static_cast<float>(static_cast<int>(i % 5) - 2) * 0.05f;
			for (size_t i = 0; i < bigBias.size(); ++i)
				bigBias[i] = static_cast<float>(i) * 0.01f;

			// Reference: naive triple loop using float accumulation (matching kernel precision)
			for (unsigned int t = 0; t < bigT; ++t)
				for (unsigned int o = 0; o < bigO; ++o)
				{
					float acc = bigBias[o];
					for (unsigned int k = 0; k < bigK; ++k)
						acc += bigX[t * bigK + k] * bigW[o * bigK + k];
					refY[t * bigO + o] = acc;
				}

			glades::transformer_kernels::gemm_rowmajor_ABt_bias(&bigX[0], bigT, bigK, &bigW[0], &bigBias[0], bigO, bigO, &bigY[0]);

			double maxErr = 0.0;
			for (size_t i = 0; i < refY.size(); ++i)
			{
				const double err = fabs(static_cast<double>(bigY[i]) - static_cast<double>(refY[i]));
				if (err > maxErr) maxErr = err;
			}
			G_assert(__FILE__, __LINE__, "==============GEMM large tiled max error too high==============", maxErr < 0.01);
		}
	}

	// ===== 4. Dropout kernels =====
	printf("-----------------------------------\n");
	printf("Dropout mask generation and application\n");
	printf("-----------------------------------\n");
	{
		glades::rng::Engine eng;
		glades::rng::seed_engine(eng, 42u);

		const size_t N = 10000;
		std::vector<unsigned char> mask(N);

		// Rate=0: all kept
		glades::transformer_kernels::generate_dropout_mask(eng, &mask[0], N, 0.0f);
		{
			size_t kept = 0;
			for (size_t i = 0; i < N; ++i)
				if (mask[i]) ++kept;
			G_assert(__FILE__, __LINE__, "==============Dropout: rate=0 should keep all==============", kept == N);
		}

		// Rate=1: all dropped
		glades::transformer_kernels::generate_dropout_mask(eng, &mask[0], N, 1.0f);
		{
			size_t kept = 0;
			for (size_t i = 0; i < N; ++i)
				if (mask[i]) ++kept;
			G_assert(__FILE__, __LINE__, "==============Dropout: rate=1 should drop all==============", kept == 0);
		}

		// Rate=0.3: roughly 70% kept
		glades::transformer_kernels::generate_dropout_mask(eng, &mask[0], N, 0.3f);
		{
			size_t kept = 0;
			for (size_t i = 0; i < N; ++i)
				if (mask[i]) ++kept;
			const double keepRatio = static_cast<double>(kept) / static_cast<double>(N);
			// Should be around 0.7 +/- 0.03
			G_assert(__FILE__, __LINE__, "==============Dropout: rate=0.3 keep ratio out of range==============",
			         keepRatio > 0.65 && keepRatio < 0.75);
		}

		// apply_dropout_mask_inplace: verify zeroed elements and scale
		{
			const size_t M = 8;
			float data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
			unsigned char m[8] = {1, 0, 1, 0, 1, 1, 0, 1};
			const float scale = 2.0f;
			glades::transformer_kernels::apply_dropout_mask_inplace(data, m, scale, M);
			G_assert(__FILE__, __LINE__, "==============Dropout apply: kept[0] wrong==============", fabs(data[0] - 2.0f) < 1e-6f);
			G_assert(__FILE__, __LINE__, "==============Dropout apply: dropped[1] not zero==============", data[1] == 0.0f);
			G_assert(__FILE__, __LINE__, "==============Dropout apply: kept[2] wrong==============", fabs(data[2] - 6.0f) < 1e-6f);
			G_assert(__FILE__, __LINE__, "==============Dropout apply: dropped[3] not zero==============", data[3] == 0.0f);
			G_assert(__FILE__, __LINE__, "==============Dropout apply: kept[4] wrong==============", fabs(data[4] - 10.0f) < 1e-6f);
			G_assert(__FILE__, __LINE__, "==============Dropout apply: kept[5] wrong==============", fabs(data[5] - 12.0f) < 1e-6f);
			G_assert(__FILE__, __LINE__, "==============Dropout apply: dropped[6] not zero==============", data[6] == 0.0f);
			G_assert(__FILE__, __LINE__, "==============Dropout apply: kept[7] wrong==============", fabs(data[7] - 16.0f) < 1e-6f);
		}
	}

	// ===== 5. Vectorized activation forward/backward =====
	printf("-----------------------------------\n");
	printf("Vectorized activations (GELU, SiLU)\n");
	printf("-----------------------------------\n");
	{
		const size_t N = 64;
		std::vector<float> x(N);
		for (size_t i = 0; i < N; ++i)
			x[i] = static_cast<float>(i) * 0.1f - 3.0f;  // range [-3, 3.3]

		// GELU forward: compare buf API with scalar reference
		{
			std::vector<float> yBuf(N, 0.0f);
			std::vector<float> yRef(N, 0.0f);
			glades::transformer_kernels::gelu_forward_buf(&x[0], &yBuf[0], N);
			for (size_t i = 0; i < N; ++i)
				yRef[i] = glades::transformer_ops::gelu(x[i]);

			double maxErr = 0.0;
			for (size_t i = 0; i < N; ++i)
			{
				const double err = fabs(static_cast<double>(yBuf[i]) - static_cast<double>(yRef[i]));
				if (err > maxErr) maxErr = err;
			}
			G_assert(__FILE__, __LINE__, "==============GELU forward: buf vs scalar mismatch==============", maxErr < 1e-6);
		}

		// GELU backward: compare buf API with scalar reference
		{
			std::vector<float> dActBuf(N);
			std::vector<float> dActRef(N);
			for (size_t i = 0; i < N; ++i)
			{
				dActBuf[i] = 1.0f;  // upstream gradient = 1 for easy comparison
				dActRef[i] = 1.0f;
			}
			glades::transformer_kernels::gelu_backward_buf(&x[0], &dActBuf[0], N);
			for (size_t i = 0; i < N; ++i)
				dActRef[i] *= glades::transformer_ops::gelu_deriv(x[i]);

			double maxErr = 0.0;
			for (size_t i = 0; i < N; ++i)
			{
				const double err = fabs(static_cast<double>(dActBuf[i]) - static_cast<double>(dActRef[i]));
				if (err > maxErr) maxErr = err;
			}
			G_assert(__FILE__, __LINE__, "==============GELU backward: buf vs scalar mismatch==============", maxErr < 1e-6);
		}

		// SiLU forward
		{
			std::vector<float> yBuf(N, 0.0f);
			std::vector<float> yRef(N, 0.0f);
			glades::transformer_kernels::silu_forward_buf(&x[0], &yBuf[0], N);
			for (size_t i = 0; i < N; ++i)
				yRef[i] = glades::transformer_ops::silu(x[i]);

			double maxErr = 0.0;
			for (size_t i = 0; i < N; ++i)
			{
				const double err = fabs(static_cast<double>(yBuf[i]) - static_cast<double>(yRef[i]));
				if (err > maxErr) maxErr = err;
			}
			G_assert(__FILE__, __LINE__, "==============SiLU forward: buf vs scalar mismatch==============", maxErr < 1e-5);
		}

		// SiLU backward
		{
			std::vector<float> dActBuf(N);
			std::vector<float> dActRef(N);
			for (size_t i = 0; i < N; ++i)
			{
				dActBuf[i] = 1.0f;
				dActRef[i] = 1.0f;
			}
			glades::transformer_kernels::silu_backward_buf(&x[0], &dActBuf[0], N);
			for (size_t i = 0; i < N; ++i)
				dActRef[i] *= glades::transformer_ops::silu_deriv(x[i]);

			double maxErr = 0.0;
			for (size_t i = 0; i < N; ++i)
			{
				const double err = fabs(static_cast<double>(dActBuf[i]) - static_cast<double>(dActRef[i]));
				if (err > maxErr) maxErr = err;
			}
			G_assert(__FILE__, __LINE__, "==============SiLU backward: buf vs scalar mismatch==============", maxErr < 1e-5);
		}
	}

	// ===== 6. Final LN integration: training produces finite loss and weight updates =====
	printf("-----------------------------------\n");
	printf("Final LN: training with RMSNorm\n");
	printf("-----------------------------------\n");
	{
		SmallTransformerSetup s(17u, 16u, 4u, 32u, 2026u);
		s.net->getTrainingConfigMutable().transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;

		G_assert(__FILE__, __LINE__, "==============FinalLN RMSNorm: init failed==============", s.net->test(s.di).ok());

		// Record logits before training
		std::vector<unsigned int> probe;
		probe.push_back(1u);
		probe.push_back(2u);
		probe.push_back(3u);
		std::vector<float> logitsBefore;
		G_assert(__FILE__, __LINE__, "==============FinalLN RMSNorm: forward before failed==============",
		         s.net->transformerLmForwardLastLogits(probe, logitsBefore).ok());
		for (size_t i = 0; i < logitsBefore.size(); ++i)
			G_assert(__FILE__, __LINE__, "==============FinalLN RMSNorm: non-finite logit before==============", std::isfinite(logitsBefore[i]));

		// Train
		const glades::NNetworkStatus st = s.net->train(s.di);
		G_assert(__FILE__, __LINE__, "==============FinalLN RMSNorm: train failed==============", st.ok());

		// Verify logits changed (parameters were updated)
		std::vector<float> logitsAfter;
		G_assert(__FILE__, __LINE__, "==============FinalLN RMSNorm: forward after failed==============",
		         s.net->transformerLmForwardLastLogits(probe, logitsAfter).ok());
		double maxDiff = 0.0;
		for (size_t i = 0; i < logitsAfter.size(); ++i)
		{
			G_assert(__FILE__, __LINE__, "==============FinalLN RMSNorm: non-finite logit after==============", std::isfinite(logitsAfter[i]));
			const double d = fabs(static_cast<double>(logitsAfter[i]) - static_cast<double>(logitsBefore[i]));
			if (d > maxDiff) maxDiff = d;
		}
		G_assert(__FILE__, __LINE__, "==============FinalLN RMSNorm: params not updated after training==============", maxDiff > 1e-6);
	}

	// ===== 7. Final LN with LayerNorm variant =====
	printf("-----------------------------------\n");
	printf("Final LN: training with LayerNorm\n");
	printf("-----------------------------------\n");
	{
		SmallTransformerSetup s(17u, 12u, 3u, 24u, 9999u);
		s.net->getTrainingConfigMutable().transformer.normType = glades::TransformerRunConfig::NORM_LAYERNORM;
		s.net->getTrainingConfigMutable().transformer.ffnKind = glades::TransformerRunConfig::FFN_MLP;
		s.net->getTrainingConfigMutable().transformer.ffnActivation = glades::TransformerRunConfig::FFN_GELU;
		s.net->getTrainingConfigMutable().transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_SINUSOIDAL;

		G_assert(__FILE__, __LINE__, "==============FinalLN LayerNorm: init failed==============", s.net->test(s.di).ok());

		std::vector<unsigned int> probe;
		probe.push_back(1u);
		probe.push_back(2u);
		std::vector<float> logitsBefore;
		G_assert(__FILE__, __LINE__, "==============FinalLN LayerNorm: forward before failed==============",
		         s.net->transformerLmForwardLastLogits(probe, logitsBefore).ok());

		const glades::NNetworkStatus st = s.net->train(s.di);
		G_assert(__FILE__, __LINE__, "==============FinalLN LayerNorm: train failed==============", st.ok());

		std::vector<float> logitsAfter;
		G_assert(__FILE__, __LINE__, "==============FinalLN LayerNorm: forward after failed==============",
		         s.net->transformerLmForwardLastLogits(probe, logitsAfter).ok());
		double maxDiff = 0.0;
		for (size_t i = 0; i < logitsAfter.size(); ++i)
		{
			const double d = fabs(static_cast<double>(logitsAfter[i]) - static_cast<double>(logitsBefore[i]));
			if (d > maxDiff) maxDiff = d;
		}
		G_assert(__FILE__, __LINE__, "==============FinalLN LayerNorm: params not updated==============", maxDiff > 1e-6);
	}

	// ===== 8. KV-cache parity still holds with final LN =====
	printf("-----------------------------------\n");
	printf("Final LN: KV-cache parity (full vs incremental)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 17u;
		const unsigned int padTokenId = vocab - 1u;
		const unsigned int T = 5u;
		std::vector<unsigned int> toks;
		for (unsigned int i = 0; i < T; ++i)
			toks.push_back(1u + i);

		TestTokenIdInput* di = new TestTokenIdInput();
		di->setTrainTokens(toks, static_cast<int>(padTokenId));
		di->mirrorTrainToTest();

		glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(16, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(static_cast<int>(vocab), glades::OutputLayerInfo::CLASSIFICATION);
		glades::NNInfo* info = new glades::NNInfo("ut_finln_kv_parity", in, hidden, out);

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
		}

		G_assert(__FILE__, __LINE__, "==============KVParity FinalLN: init failed==============", net.test(di).ok());
		glades::NNetwork::TransformerLmSession session;
		G_assert(__FILE__, __LINE__, "==============KVParity FinalLN: session reset failed==============", net.transformerLmSessionReset(session, T).ok());

		std::vector<unsigned int> prefix;
		std::vector<float> logitsFull;
		std::vector<float> logitsKv;
		for (unsigned int t = 0; t < T; ++t)
		{
			prefix.push_back(toks[t]);
			G_assert(__FILE__, __LINE__, "==============KVParity FinalLN: ForwardLastLogits failed==============",
			         net.transformerLmForwardLastLogits(prefix, logitsFull).ok());
			G_assert(__FILE__, __LINE__, "==============KVParity FinalLN: SessionAppend failed==============",
			         net.transformerLmSessionAppend(session, toks[t], &logitsKv).ok());
			G_assert(__FILE__, __LINE__, "==============KVParity FinalLN: logits size mismatch==============",
			         logitsFull.size() == vocab && logitsKv.size() == vocab);
			double maxAbs = 0.0;
			for (unsigned int i = 0; i < vocab; ++i)
			{
				const double d = fabs(static_cast<double>(logitsFull[i]) - static_cast<double>(logitsKv[i]));
				if (d > maxAbs) maxAbs = d;
				G_assert(__FILE__, __LINE__, "==============KVParity FinalLN: non-finite logit==============",
				         std::isfinite(logitsFull[i]) && std::isfinite(logitsKv[i]));
			}
			G_assert(__FILE__, __LINE__, "==============KVParity FinalLN: logits diverge==============", maxAbs < 1e-3);
		}

		delete di;
		delete info;
	}

	// ===== 9. Training with dropout > 0 still converges =====
	printf("-----------------------------------\n");
	printf("Dropout integration: training with dropout rates > 0\n");
	printf("-----------------------------------\n");
	{
		SmallTransformerSetup s(17u, 16u, 4u, 32u, 777u);
		glades::TrainingConfig& cfg = s.net->getTrainingConfigMutable();
		cfg.transformer.embeddingDropoutRate = 0.1f;
		cfg.transformer.residualDropoutRate = 0.1f;

		G_assert(__FILE__, __LINE__, "==============Dropout integration: init failed==============", s.net->test(s.di).ok());

		// Record logits before training
		std::vector<unsigned int> probe;
		probe.push_back(1u);
		probe.push_back(2u);
		probe.push_back(3u);
		std::vector<float> logitsBefore;
		G_assert(__FILE__, __LINE__, "==============Dropout integration: forward before failed==============",
		         s.net->transformerLmForwardLastLogits(probe, logitsBefore).ok());

		const glades::NNetworkStatus st = s.net->train(s.di);
		G_assert(__FILE__, __LINE__, "==============Dropout integration: train failed==============", st.ok());

		// Verify params were updated despite dropout
		std::vector<float> logitsAfter;
		G_assert(__FILE__, __LINE__, "==============Dropout integration: forward after failed==============",
		         s.net->transformerLmForwardLastLogits(probe, logitsAfter).ok());
		double maxDiff = 0.0;
		for (size_t i = 0; i < logitsAfter.size(); ++i)
		{
			G_assert(__FILE__, __LINE__, "==============Dropout integration: non-finite logit==============", std::isfinite(logitsAfter[i]));
			const double d = fabs(static_cast<double>(logitsAfter[i]) - static_cast<double>(logitsBefore[i]));
			if (d > maxDiff) maxDiff = d;
		}
		G_assert(__FILE__, __LINE__, "==============Dropout integration: params not updated==============", maxDiff > 1e-6);
	}

	// ===== 10. Checkpoint round-trip: save and load preserves final LN tensors =====
	printf("-----------------------------------\n");
	printf("Checkpoint persistence: final LN save/load round-trip\n");
	printf("-----------------------------------\n");
	{
		const unsigned int vocab = 17u;
		const unsigned int dModel = 16u;
		SmallTransformerSetup s(vocab, dModel, 4u, 32u, 1234u);
		G_assert(__FILE__, __LINE__, "==============Checkpoint: init failed==============", s.net->test(s.di).ok());
		G_assert(__FILE__, __LINE__, "==============Checkpoint: train failed==============", s.net->train(s.di).ok());

		// Save model
		G_assert(__FILE__, __LINE__, "==============Checkpoint: save failed==============",
		         s.net->saveModel("ut_transformer_improvements_ckpt").ok());

		// Record logits from trained network
		std::vector<unsigned int> prefix;
		prefix.push_back(1u);
		prefix.push_back(2u);
		prefix.push_back(3u);
		std::vector<float> logits1;
		G_assert(__FILE__, __LINE__, "==============Checkpoint: forward1 failed==============",
		         s.net->transformerLmForwardLastLogits(prefix, logits1).ok());

		// Create a new network and load the checkpoint
		SmallTransformerSetup s2(vocab, dModel, 4u, 32u, 9999u); // different seed
		G_assert(__FILE__, __LINE__, "==============Checkpoint: load init failed==============", s2.net->test(s2.di).ok());
		G_assert(__FILE__, __LINE__, "==============Checkpoint: load failed==============",
		         s2.net->loadModel("ut_transformer_improvements_ckpt", s2.di).ok());

		// Verify logits match between saved and loaded networks
		// (this implicitly verifies all weights including final LN gamma/beta were restored)
		std::vector<float> logits2;
		G_assert(__FILE__, __LINE__, "==============Checkpoint: forward2 failed==============",
		         s2.net->transformerLmForwardLastLogits(prefix, logits2).ok());
		G_assert(__FILE__, __LINE__, "==============Checkpoint: logits size mismatch==============",
		         logits1.size() == logits2.size());
		{
			double maxErr = 0.0;
			for (size_t i = 0; i < logits1.size(); ++i)
			{
				G_assert(__FILE__, __LINE__, "==============Checkpoint: non-finite logit==============",
				         std::isfinite(logits1[i]) && std::isfinite(logits2[i]));
				const double e = fabs(static_cast<double>(logits1[i]) - static_cast<double>(logits2[i]));
				if (e > maxErr) maxErr = e;
			}
			G_assert(__FILE__, __LINE__, "==============Checkpoint: logits differ after save/load==============", maxErr < 1e-4);
		}
	}

	// ===== 11. Multi-layer transformer with GELU + LayerNorm + dropout =====
	printf("-----------------------------------\n");
	printf("Multi-layer: GELU + LayerNorm + dropout training\n");
	printf("-----------------------------------\n");
	{
		SmallTransformerSetup s(17u, 12u, 3u, 24u, 5555u, 2);
		glades::TrainingConfig& cfg = s.net->getTrainingConfigMutable();
		cfg.transformer.normType = glades::TransformerRunConfig::NORM_LAYERNORM;
		cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_MLP;
		cfg.transformer.ffnActivation = glades::TransformerRunConfig::FFN_GELU;
		cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_SINUSOIDAL;
		cfg.transformer.embeddingDropoutRate = 0.05f;
		cfg.transformer.residualDropoutRate = 0.05f;

		G_assert(__FILE__, __LINE__, "==============MultiLayer: init failed==============", s.net->test(s.di).ok());
		const glades::NNetworkStatus st = s.net->train(s.di);
		G_assert(__FILE__, __LINE__, "==============MultiLayer: train failed==============", st.ok());
	}

	// ===== 12. Determinism: same seed produces identical logits =====
	printf("-----------------------------------\n");
	printf("Determinism: same seed same config same logits\n");
	printf("-----------------------------------\n");
	{
		std::vector<unsigned int> probe;
		probe.push_back(1u);
		probe.push_back(2u);
		probe.push_back(3u);

		SmallTransformerSetup a(17u, 16u, 4u, 32u, 42u);
		G_assert(__FILE__, __LINE__, "==============Determinism: init A failed==============", a.net->test(a.di).ok());
		std::vector<float> logitsA;
		G_assert(__FILE__, __LINE__, "==============Determinism: forward A failed==============",
		         a.net->transformerLmForwardLastLogits(probe, logitsA).ok());

		SmallTransformerSetup b(17u, 16u, 4u, 32u, 42u);
		G_assert(__FILE__, __LINE__, "==============Determinism: init B failed==============", b.net->test(b.di).ok());
		std::vector<float> logitsB;
		G_assert(__FILE__, __LINE__, "==============Determinism: forward B failed==============",
		         b.net->transformerLmForwardLastLogits(probe, logitsB).ok());

		G_assert(__FILE__, __LINE__, "==============Determinism: size mismatch==============", logitsA.size() == logitsB.size());
		for (size_t i = 0; i < logitsA.size(); ++i)
			G_assert(__FILE__, __LINE__, "==============Determinism: logits differ==============",
			         fabs(static_cast<double>(logitsA[i]) - static_cast<double>(logitsB[i])) < 1e-7);
	}

	// ===== 13. TokenInput file import works on the real token LM path =====
	printf("-----------------------------------\n");
	printf("TokenInput: file import feeds transformer training/inference\n");
	printf("-----------------------------------\n");
	{
		const char* path = "/tmp/ut_transformer_improvements_tokeninput.txt";
		std::ofstream out(path);
		out << "1 2 3 4 5\n";
		out << "2 3 4 5 6\n";
		out.close();

		glades::TokenInput di;
		di.setPadTokenId(16);
		di.setMirrorTrainToTestOnImplicitSplit(true);
		di.import(shmea::GString(path));
		G_assert(__FILE__, __LINE__, "==============TokenInput improvements: import failed==============", di.loadedOk());
		G_assert(__FILE__, __LINE__, "==============TokenInput improvements: train split empty==============", di.getTrainSize() > 0u);
		G_assert(__FILE__, __LINE__, "==============TokenInput improvements: test split empty==============", di.getTestSize() > 0u);

		glades::InputLayerInfo* in = new glades::InputLayerInfo(1, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f);
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(16, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::LINEAR, 1.0f));
		glades::OutputLayerInfo* outInfo = new glades::OutputLayerInfo(17, glades::OutputLayerInfo::CLASSIFICATION);

		glades::NNInfo* info = new glades::NNInfo("ut_transformer_improvements_tokeninput", in, hidden, outInfo);
		glades::NNetwork* net = new glades::NNetwork(info, glades::NNetwork::TYPE_TRANSFORMER_DECODER);
		net->setSeed(31337u);
		net->getTerminatorMutable().setEpoch(1);
		net->getTerminatorMutable().setAccuracy(0);

		glades::TrainingConfig& cfg = net->getTrainingConfigMutable();
		cfg.transformer.enableTokenEmbedding = true;
		cfg.transformer.vocabSizeOverride = 17;
		cfg.transformer.tieEmbeddings = true;
		cfg.transformer.padTokenId = 16;
		cfg.transformer.nHeadsOverride = 4;
		cfg.transformer.nKVHeadsOverride = 4;
		cfg.transformer.dFFOverride = 32;
		cfg.transformer.ffnKind = glades::TransformerRunConfig::FFN_SWIGLU;
		cfg.transformer.normType = glades::TransformerRunConfig::NORM_RMSNORM;
		cfg.transformer.positionalEncoding = glades::TransformerRunConfig::POSENC_ROPE;
		cfg.transformer.ropeTheta = 10000.0f;
		cfg.optimizer.type = glades::OptimizerConfig::ADAMW;

		G_assert(__FILE__, __LINE__, "==============TokenInput improvements: init failed==============", net->test(&di).ok());
		G_assert(__FILE__, __LINE__, "==============TokenInput improvements: train failed==============", net->train(&di).ok());

		std::vector<unsigned int> probe;
		probe.push_back(1u);
		probe.push_back(2u);
		probe.push_back(3u);
		std::vector<float> logits;
		G_assert(__FILE__, __LINE__, "==============TokenInput improvements: forward failed==============",
		         net->transformerLmForwardLastLogits(probe, logits).ok());
		G_assert(__FILE__, __LINE__, "==============TokenInput improvements: logits size mismatch==============", logits.size() == 17u);

		delete net;
		delete info;
	}

	printf("\n============================================================\n");
}
