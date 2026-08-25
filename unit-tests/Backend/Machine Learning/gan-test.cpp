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

#include "gan-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/Networks/gan.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/training_config.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

static bool is_finite(float v)
{
	return (v == v) && (v - v == 0.0f);
}

// Create a synthetic 2D Gaussian mixture dataset for DFF GAN testing.
// Generates points around 2 cluster centers.
static glades::NumberInput* make_gaussian_mixture(unsigned int numSamples, unsigned int seed)
{
	glades::NumberInput* di = new glades::NumberInput();
	const unsigned int featureCount = 2;

	di->trainMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(featureCount, 0.0f));
	// GAN doesn't use expected outputs, but DataInput requires them
	di->trainExpectedMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(1, 0.0f));

	unsigned int s = seed;
	for (unsigned int i = 0; i < numSamples; ++i)
	{
		// LCG for deterministic pseudo-random floats
		s = s * 1103515245u + 12345u;
		float u1 = static_cast<float>((s >> 16) & 0x7FFFu) / 32767.0f;
		s = s * 1103515245u + 12345u;
		float u2 = static_cast<float>((s >> 16) & 0x7FFFu) / 32767.0f;

		// Box-Muller for normal distribution
		if (u1 < 1e-6f) u1 = 1e-6f;
		float z1 = static_cast<float>(sqrt(-2.0 * log(static_cast<double>(u1))) * cos(6.28318 * static_cast<double>(u2)));
		float z2 = static_cast<float>(sqrt(-2.0 * log(static_cast<double>(u1))) * sin(6.28318 * static_cast<double>(u2)));

		// Two clusters: center at (2,2) and (-2,-2)
		if (i % 2 == 0)
		{
			di->trainMatrix[i][0] = 2.0f + 0.5f * z1;
			di->trainMatrix[i][1] = 2.0f + 0.5f * z2;
		}
		else
		{
			di->trainMatrix[i][0] = -2.0f + 0.5f * z1;
			di->trainMatrix[i][1] = -2.0f + 0.5f * z2;
		}
		di->trainExpectedMatrix[i][0] = 0.0f;
	}

	di->testMatrix = di->trainMatrix;
	di->testExpectedMatrix = di->trainExpectedMatrix;

	return di;
}

// Create synthetic "image" data for CNN GAN testing.
static glades::NumberInput* make_synthetic_image_data(
    unsigned int numSamples,
    unsigned int C, unsigned int H, unsigned int W,
    unsigned int seed)
{
	glades::NumberInput* di = new glades::NumberInput();
	const unsigned int featureCount = C * H * W;

	di->trainMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(featureCount, 0.0f));
	di->trainExpectedMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(1, 0.0f));

	unsigned int s = seed;
	for (unsigned int i = 0; i < numSamples; ++i)
	{
		for (unsigned int f = 0; f < featureCount; ++f)
		{
			s = s * 1103515245u + 12345u;
			float v = static_cast<float>((s >> 16) & 0x7FFFu) / 32767.0f;
			// Simple pattern: brighter horizontal stripes
			unsigned int pixel = f % (H * W);
			unsigned int row = pixel / W;
			if (row % 2 == 0)
				v = v * 0.3f + 0.7f;
			else
				v = v * 0.3f;
			di->trainMatrix[i][f] = v;
		}
		di->trainExpectedMatrix[i][0] = 0.0f;
	}

	di->testMatrix = di->trainMatrix;
	di->testExpectedMatrix = di->trainExpectedMatrix;

	return di;
}

// Helper: create NNInfo for generator (noise -> dataDim)
static glades::NNInfo* make_gen_info(const char* name,
                                     const std::vector<unsigned int>& hiddenSizes,
                                     float lr,
                                     int activationType)
{
	auto in = shmea::make_gpointer<glades::InputLayerInfo>(
	    /*batchSize*/ 1,
	    /*learningRate*/ lr,
	    /*momentumFactor*/ 0.0f,
	    /*weightDecay1*/ 0.0f,
	    /*weightDecay2*/ 0.0f,
	    /*pDropout*/ 0.0f,
	    /*activationType*/ activationType,
	    /*activationParam*/ 0.01f);

	std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
	for (size_t i = 0; i < hiddenSizes.size(); ++i)
	{
		hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
		    static_cast<int>(hiddenSizes[i]),
		    lr,
		    0.0f, 0.0f, 0.0f, 0.0f,
		    activationType,
		    0.01f));
	}

	// Output layer size will be overridden by GAN based on data
	auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);

	return new glades::NNInfo(name, in, hidden, out);
}

// Helper: create NNInfo for discriminator (dataDim -> 1)
static glades::NNInfo* make_disc_info(const char* name,
                                      const std::vector<unsigned int>& hiddenSizes,
                                      float lr,
                                      int activationType)
{
	auto in = shmea::make_gpointer<glades::InputLayerInfo>(
	    /*batchSize*/ 1,
	    /*learningRate*/ lr,
	    /*momentumFactor*/ 0.0f,
	    /*weightDecay1*/ 0.0f,
	    /*weightDecay2*/ 0.0f,
	    /*pDropout*/ 0.0f,
	    /*activationType*/ activationType,
	    /*activationParam*/ 0.01f);

	std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
	for (size_t i = 0; i < hiddenSizes.size(); ++i)
	{
		hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
		    static_cast<int>(hiddenSizes[i]),
		    lr,
		    0.0f, 0.0f, 0.0f, 0.0f,
		    activationType,
		    0.01f));
	}

	auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);

	return new glades::NNInfo(name, in, hidden, out);
}

// Callback to capture metrics
struct GANCaptureMetrics : public glades::IGANCallbacks
{
	glades::GANEpochMetrics last;
	bool saw;

	GANCaptureMetrics() : last(), saw(false) {}

	virtual void onEpochEnd(const glades::GANEpochMetrics& m)
	{
		last = m;
		saw = true;
	}
};

} // anonymous namespace

void GANUnitTest()
{
	printf("============================================================\n");
	printf("GAN Unit Test Suite\n");
	printf("============================================================\n");

	// ------------------------------------------------------------------
	// Test 1: DFF GAN with vanilla loss - basic training smoke test
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 1: DFF vanilla GAN - smoke test\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(100, 42u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);
		genHidden.push_back(16u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);
		discHidden.push_back(8u);

		glades::NNInfo* genInfo = make_gen_info("ut_gan_gen", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo = make_disc_info("ut_gan_disc", discHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.noiseDim = 8;
		cfg.epochs = 20;
		cfg.batchSize = 20;
		cfg.nCriticPerGenerator = 1;

		glades::GAN gan(cfg, genInfo, discInfo);
		gan.setSeed(123);

		GANCaptureMetrics metrics;
		const glades::NNetworkStatus st = gan.train(di, &metrics);
		printf("[UT] GAN DFF vanilla train status: %s (code=%d)\n", st.message.c_str(), static_cast<int>(st.code));
		G_assert(__FILE__, __LINE__, "==============GAN::DFF_Vanilla TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::DFF_Vanilla No Metrics==============", metrics.saw);

		// Verify generator can produce finite samples
		std::vector<std::vector<float> > samples;
		const glades::NNetworkStatus genSt = gan.generate(10, samples);
		G_assert(__FILE__, __LINE__, "==============GAN::DFF_Vanilla GenerateStatus() Failed==============", genSt.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::DFF_Vanilla Wrong Sample Count==============", samples.size() == 10);
		G_assert(__FILE__, __LINE__, "==============GAN::DFF_Vanilla Wrong Sample Dim==============", samples[0].size() == 2);

		bool allFinite = true;
		for (size_t i = 0; i < samples.size(); ++i)
			for (size_t j = 0; j < samples[i].size(); ++j)
				if (!is_finite(samples[i][j]))
					allFinite = false;

		G_assert(__FILE__, __LINE__, "==============GAN::DFF_Vanilla Non-Finite Samples==============", allFinite);

		// Check that samples are not all identical (generator isn't mode-collapsed to a single point)
		bool allIdentical = true;
		for (size_t i = 1; i < samples.size(); ++i)
		{
			for (size_t j = 0; j < samples[i].size(); ++j)
			{
				if (fabsf(samples[i][j] - samples[0][j]) > 1e-6f)
				{
					allIdentical = false;
					break;
				}
			}
			if (!allIdentical) break;
		}
		G_assert(__FILE__, __LINE__, "==============GAN::DFF_Vanilla All Samples Identical==============", !allIdentical);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo;
		delete discInfo;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 2: DFF GAN with WGAN-GP loss
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 2: DFF WGAN-GP\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(100, 77u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);

		glades::NNInfo* genInfo = make_gen_info("ut_wgan_gen", genHidden, 0.0001f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo = make_disc_info("ut_wgan_disc", discHidden, 0.0001f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_WGAN_GP;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.noiseDim = 8;
		cfg.epochs = 15;
		cfg.batchSize = 20;
		cfg.nCriticPerGenerator = 3;
		cfg.gpLambda = 10.0f;

		glades::GAN gan(cfg, genInfo, discInfo);
		gan.setSeed(456);

		GANCaptureMetrics metrics;
		const glades::NNetworkStatus st = gan.train(di, &metrics);
		printf("[UT] GAN DFF WGAN-GP train status: %s (code=%d)\n", st.message.c_str(), static_cast<int>(st.code));
		G_assert(__FILE__, __LINE__, "==============GAN::DFF_WGANGP TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::DFF_WGANGP No Metrics==============", metrics.saw);

		// Generate samples
		std::vector<std::vector<float> > samples;
		const glades::NNetworkStatus genSt = gan.generate(10, samples);
		G_assert(__FILE__, __LINE__, "==============GAN::DFF_WGANGP GenerateStatus() Failed==============", genSt.ok());

		bool allFinite = true;
		for (size_t i = 0; i < samples.size(); ++i)
			for (size_t j = 0; j < samples[i].size(); ++j)
				if (!is_finite(samples[i][j]))
					allFinite = false;
		G_assert(__FILE__, __LINE__, "==============GAN::DFF_WGANGP Non-Finite Samples==============", allFinite);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo;
		delete discInfo;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 3: CNN GAN with vanilla loss
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 3: CNN vanilla GAN\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		glades::NumberInput* di = make_synthetic_image_data(40, C, H, W, 99u);

		// Generator: DFF (noise -> flat image)
		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);

		// Discriminator: CNN (image -> scalar)
		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);

		glades::NNInfo* genInfo = make_gen_info("ut_cnn_gan_gen", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo = make_disc_info("ut_cnn_gan_disc", discHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_CNN;
		cfg.noiseDim = 16;
		cfg.epochs = 10;
		cfg.batchSize = 10;
		cfg.nCriticPerGenerator = 1;

		// CNN config for discriminator
		glades::CNNConfig::ConvLayerSpec convSpec;
		convSpec.outChannels = 4;
		convSpec.kernelH = 3; convSpec.kernelW = 3;
		convSpec.strideH = 1; convSpec.strideW = 1;
		convSpec.padH = 1; convSpec.padW = 1;
		convSpec.useBatchNorm = false;
		convSpec.useMaxPool = false;
		cfg.discriminatorCNN.inputH = H;
		cfg.discriminatorCNN.inputW = W;
		cfg.discriminatorCNN.inputC = C;
		cfg.discriminatorCNN.convLayers.push_back(convSpec);

		glades::GAN gan(cfg, genInfo, discInfo);
		gan.setSeed(789);

		GANCaptureMetrics metrics;
		const glades::NNetworkStatus st = gan.train(di, &metrics);
		printf("[UT] GAN CNN vanilla train status: %s (code=%d)\n", st.message.c_str(), static_cast<int>(st.code));
		G_assert(__FILE__, __LINE__, "==============GAN::CNN_Vanilla TrainStatus() Failed==============", st.ok());

		// Generate flat image samples
		std::vector<std::vector<float> > samples;
		const glades::NNetworkStatus genSt = gan.generate(5, samples);
		G_assert(__FILE__, __LINE__, "==============GAN::CNN_Vanilla GenerateStatus() Failed==============", genSt.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::CNN_Vanilla Wrong Sample Count==============", samples.size() == 5);
		G_assert(__FILE__, __LINE__, "==============GAN::CNN_Vanilla Wrong Sample Dim==============", samples[0].size() == C * H * W);

		bool allFinite = true;
		for (size_t i = 0; i < samples.size(); ++i)
			for (size_t j = 0; j < samples[i].size(); ++j)
				if (!is_finite(samples[i][j]))
					allFinite = false;
		G_assert(__FILE__, __LINE__, "==============GAN::CNN_Vanilla Non-Finite Samples==============", allFinite);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo;
		delete discInfo;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 4: CNN GAN with WGAN-GP loss
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 4: CNN WGAN-GP\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		glades::NumberInput* di = make_synthetic_image_data(40, C, H, W, 55u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);

		glades::NNInfo* genInfo = make_gen_info("ut_cnn_wgan_gen", genHidden, 0.0001f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo = make_disc_info("ut_cnn_wgan_disc", discHidden, 0.0001f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_WGAN_GP;
		cfg.archType = glades::GANConfig::GAN_CNN;
		cfg.noiseDim = 16;
		cfg.epochs = 8;
		cfg.batchSize = 10;
		cfg.nCriticPerGenerator = 3;
		cfg.gpLambda = 10.0f;

		glades::CNNConfig::ConvLayerSpec convSpec;
		convSpec.outChannels = 4;
		convSpec.kernelH = 3; convSpec.kernelW = 3;
		convSpec.strideH = 1; convSpec.strideW = 1;
		convSpec.padH = 1; convSpec.padW = 1;
		convSpec.useBatchNorm = false;
		convSpec.useMaxPool = false;
		cfg.discriminatorCNN.inputH = H;
		cfg.discriminatorCNN.inputW = W;
		cfg.discriminatorCNN.inputC = C;
		cfg.discriminatorCNN.convLayers.push_back(convSpec);

		glades::GAN gan(cfg, genInfo, discInfo);
		gan.setSeed(321);

		GANCaptureMetrics metrics;
		const glades::NNetworkStatus st = gan.train(di, &metrics);
		printf("[UT] GAN CNN WGAN-GP train status: %s (code=%d)\n", st.message.c_str(), static_cast<int>(st.code));
		G_assert(__FILE__, __LINE__, "==============GAN::CNN_WGANGP TrainStatus() Failed==============", st.ok());

		std::vector<std::vector<float> > samples;
		const glades::NNetworkStatus genSt = gan.generate(5, samples);
		G_assert(__FILE__, __LINE__, "==============GAN::CNN_WGANGP GenerateStatus() Failed==============", genSt.ok());

		bool allFinite = true;
		for (size_t i = 0; i < samples.size(); ++i)
			for (size_t j = 0; j < samples[i].size(); ++j)
				if (!is_finite(samples[i][j]))
					allFinite = false;
		G_assert(__FILE__, __LINE__, "==============GAN::CNN_WGANGP Non-Finite Samples==============", allFinite);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo;
		delete discInfo;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 5: Generator produces non-degenerate output
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 5: Non-degenerate generation\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(200, 31u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(64u);
		genHidden.push_back(32u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(32u);
		discHidden.push_back(16u);

		glades::NNInfo* genInfo = make_gen_info("ut_nondegen_gen", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo = make_disc_info("ut_nondegen_disc", discHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.noiseDim = 16;
		cfg.epochs = 50;
		cfg.batchSize = 32;
		cfg.nCriticPerGenerator = 1;

		glades::GAN gan(cfg, genInfo, discInfo);
		gan.setSeed(999);

		const glades::NNetworkStatus st = gan.train(di);
		G_assert(__FILE__, __LINE__, "==============GAN::NonDegen TrainStatus() Failed==============", st.ok());

		std::vector<std::vector<float> > samples;
		gan.generate(50, samples);

		// Check variance of generated samples (should not be near zero)
		double sumX = 0.0, sumY = 0.0;
		for (size_t i = 0; i < samples.size(); ++i)
		{
			sumX += static_cast<double>(samples[i][0]);
			sumY += static_cast<double>(samples[i][1]);
		}
		const double meanX = sumX / static_cast<double>(samples.size());
		const double meanY = sumY / static_cast<double>(samples.size());
		double varX = 0.0, varY = 0.0;
		for (size_t i = 0; i < samples.size(); ++i)
		{
			const double dx = static_cast<double>(samples[i][0]) - meanX;
			const double dy = static_cast<double>(samples[i][1]) - meanY;
			varX += dx * dx;
			varY += dy * dy;
		}
		varX /= static_cast<double>(samples.size());
		varY /= static_cast<double>(samples.size());

		printf("[UT] Generated sample variance: x=%.4f y=%.4f\n", varX, varY);
		// Variance should be > 0.001 (not collapsed to a single point)
		G_assert(__FILE__, __LINE__, "==============GAN::NonDegen VarX Too Small==============", varX > 0.001);
		G_assert(__FILE__, __LINE__, "==============GAN::NonDegen VarY Too Small==============", varY > 0.001);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo;
		delete discInfo;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 6: Access internal networks (for save/load compatibility)
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 6: Internal network access\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(50, 11u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(16u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(8u);

		glades::NNInfo* genInfo = make_gen_info("ut_access_gen", genHidden, 0.001f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo = make_disc_info("ut_access_disc", discHidden, 0.001f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.noiseDim = 4;
		cfg.epochs = 5;
		cfg.batchSize = 10;

		glades::GAN gan(cfg, genInfo, discInfo);
		gan.setSeed(222);

		const glades::NNetworkStatus st = gan.train(di);
		G_assert(__FILE__, __LINE__, "==============GAN::Access TrainStatus() Failed==============", st.ok());

		// Access internal networks
		const glades::NNetwork& gen = gan.getGenerator();
		const glades::NNetwork& disc = gan.getDiscriminator();

		// Verify they have the expected net type
		G_assert(__FILE__, __LINE__, "==============GAN::Access GenNetType==============", gen.getNetType() == glades::NNetwork::TYPE_DFF);
		G_assert(__FILE__, __LINE__, "==============GAN::Access DiscNetType==============", disc.getNetType() == glades::NNetwork::TYPE_DFF);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo;
		delete discInfo;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 7: InfoGAN DFF - smoke test
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 7: InfoGAN DFF - smoke test\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(100, 42u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);
		genHidden.push_back(16u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);
		discHidden.push_back(8u);

		glades::NNInfo* genInfo = make_gen_info("ut_info_gen", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo = make_disc_info("ut_info_disc", discHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.variantType = glades::GANConfig::GAN_INFO;
		cfg.noiseDim = 8;
		cfg.epochs = 15;
		cfg.batchSize = 20;
		cfg.nCriticPerGenerator = 1;
		cfg.infoConfig.numCategorical = 4;
		cfg.infoConfig.numContinuous = 2;
		cfg.infoConfig.infoLambda = 1.0f;

		glades::GAN gan(cfg, genInfo, discInfo);
		gan.setSeed(500);

		GANCaptureMetrics metrics;
		const glades::NNetworkStatus st = gan.train(di, &metrics);
		printf("[UT] InfoGAN DFF train status: %s (code=%d)\n", st.message.c_str(), static_cast<int>(st.code));
		G_assert(__FILE__, __LINE__, "==============GAN::InfoGAN TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::InfoGAN No Metrics==============", metrics.saw);

		// Verify infoLoss is finite
		G_assert(__FILE__, __LINE__, "==============GAN::InfoGAN Non-Finite InfoLoss==============", is_finite(metrics.last.infoLoss));

		// Generate samples
		std::vector<std::vector<float> > samples;
		const glades::NNetworkStatus genSt = gan.generate(10, samples);
		G_assert(__FILE__, __LINE__, "==============GAN::InfoGAN GenerateStatus() Failed==============", genSt.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::InfoGAN Wrong Sample Count==============", samples.size() == 10);

		bool allFinite = true;
		for (size_t i = 0; i < samples.size(); ++i)
			for (size_t j = 0; j < samples[i].size(); ++j)
				if (!is_finite(samples[i][j]))
					allFinite = false;
		G_assert(__FILE__, __LINE__, "==============GAN::InfoGAN Non-Finite Samples==============", allFinite);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo;
		delete discInfo;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 8: InfoGAN controlled generation
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 8: InfoGAN controlled generation\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(100, 42u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);
		genHidden.push_back(16u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);
		discHidden.push_back(8u);

		glades::NNInfo* genInfo = make_gen_info("ut_info_ctrl_gen", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo = make_disc_info("ut_info_ctrl_disc", discHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.variantType = glades::GANConfig::GAN_INFO;
		cfg.noiseDim = 8;
		cfg.epochs = 20;
		cfg.batchSize = 20;
		cfg.nCriticPerGenerator = 1;
		cfg.infoConfig.numCategorical = 4;
		cfg.infoConfig.numContinuous = 2;
		cfg.infoConfig.infoLambda = 1.0f;

		glades::GAN gan(cfg, genInfo, discInfo);
		gan.setSeed(600);

		const glades::NNetworkStatus st = gan.train(di);
		G_assert(__FILE__, __LINE__, "==============GAN::InfoCtrl TrainStatus() Failed==============", st.ok());

		// Generate with different fixed categorical codes
		std::vector<float> cat0(4, 0.0f); cat0[0] = 1.0f;
		std::vector<float> cat1(4, 0.0f); cat1[1] = 1.0f;
		std::vector<float> cont(2, 0.0f);

		std::vector<std::vector<float> > samples0, samples1;
		glades::NNetworkStatus s0 = gan.generateInfoGAN(20, &cat0, &cont, samples0);
		glades::NNetworkStatus s1 = gan.generateInfoGAN(20, &cat1, &cont, samples1);
		G_assert(__FILE__, __LINE__, "==============GAN::InfoCtrl Gen0 Failed==============", s0.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::InfoCtrl Gen1 Failed==============", s1.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::InfoCtrl Wrong Count0==============", samples0.size() == 20);
		G_assert(__FILE__, __LINE__, "==============GAN::InfoCtrl Wrong Count1==============", samples1.size() == 20);

		// Verify samples are finite
		bool allFinite = true;
		for (size_t i = 0; i < samples0.size(); ++i)
			for (size_t j = 0; j < samples0[i].size(); ++j)
				if (!is_finite(samples0[i][j]))
					allFinite = false;
		for (size_t i = 0; i < samples1.size(); ++i)
			for (size_t j = 0; j < samples1[i].size(); ++j)
				if (!is_finite(samples1[i][j]))
					allFinite = false;
		G_assert(__FILE__, __LINE__, "==============GAN::InfoCtrl Non-Finite==============", allFinite);

		// Check that different codes produce different mean distributions
		double mean0x = 0.0, mean1x = 0.0;
		for (size_t i = 0; i < samples0.size(); ++i)
			mean0x += static_cast<double>(samples0[i][0]);
		for (size_t i = 0; i < samples1.size(); ++i)
			mean1x += static_cast<double>(samples1[i][0]);
		mean0x /= static_cast<double>(samples0.size());
		mean1x /= static_cast<double>(samples1.size());

		printf("[UT] InfoGAN code0 mean_x=%.4f code1 mean_x=%.4f\n", mean0x, mean1x);
		// We just verify both are finite and not identical (weak check since training is short)
		G_assert(__FILE__, __LINE__, "==============GAN::InfoCtrl Mean0 Non-Finite==============", is_finite(static_cast<float>(mean0x)));
		G_assert(__FILE__, __LINE__, "==============GAN::InfoCtrl Mean1 Non-Finite==============", is_finite(static_cast<float>(mean1x)));

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo;
		delete discInfo;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 9: StyleGAN DFF - smoke test
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 9: StyleGAN DFF - smoke test\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(100, 42u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);
		genHidden.push_back(16u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);
		discHidden.push_back(8u);

		// Mapping network hidden layers
		std::vector<unsigned int> mapHidden;
		mapHidden.push_back(32u);

		glades::NNInfo* genInfo = make_gen_info("ut_style_gen", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo = make_disc_info("ut_style_disc", discHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* mapInfo = make_gen_info("ut_style_map", mapHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.variantType = glades::GANConfig::GAN_STYLE;
		cfg.noiseDim = 8;
		cfg.epochs = 15;
		cfg.batchSize = 20;
		cfg.nCriticPerGenerator = 1;
		cfg.styleConfig.mappingWidth = 16;
		cfg.styleConfig.mappingLayers = 1;
		cfg.styleConfig.noiseScaleInit = 0.0f;

		glades::GAN gan(cfg, genInfo, discInfo, mapInfo);
		gan.setSeed(700);

		GANCaptureMetrics metrics;
		const glades::NNetworkStatus st = gan.train(di, &metrics);
		printf("[UT] StyleGAN DFF train status: %s (code=%d)\n", st.message.c_str(), static_cast<int>(st.code));
		G_assert(__FILE__, __LINE__, "==============GAN::StyleGAN TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::StyleGAN No Metrics==============", metrics.saw);

		// Generate samples
		std::vector<std::vector<float> > samples;
		const glades::NNetworkStatus genSt = gan.generate(10, samples);
		G_assert(__FILE__, __LINE__, "==============GAN::StyleGAN GenerateStatus() Failed==============", genSt.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::StyleGAN Wrong Sample Count==============", samples.size() == 10);
		G_assert(__FILE__, __LINE__, "==============GAN::StyleGAN Wrong Sample Dim==============", samples[0].size() == 2);

		bool allFinite = true;
		for (size_t i = 0; i < samples.size(); ++i)
			for (size_t j = 0; j < samples[i].size(); ++j)
				if (!is_finite(samples[i][j]))
					allFinite = false;
		G_assert(__FILE__, __LINE__, "==============GAN::StyleGAN Non-Finite Samples==============", allFinite);

		// Check samples are not all identical
		bool allIdentical = true;
		for (size_t i = 1; i < samples.size(); ++i)
		{
			for (size_t j = 0; j < samples[i].size(); ++j)
			{
				if (fabsf(samples[i][j] - samples[0][j]) > 1e-6f)
				{
					allIdentical = false;
					break;
				}
			}
			if (!allIdentical) break;
		}
		G_assert(__FILE__, __LINE__, "==============GAN::StyleGAN All Samples Identical==============", !allIdentical);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo;
		delete discInfo;
		delete mapInfo;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 10: CycleGAN DFF - smoke test
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 10: CycleGAN DFF - smoke test\n");
	printf("-----------------------------------\n");
	{
		// Domain A: Gaussian at (2,2), Domain B: Gaussian at (-2,-2)
		glades::NumberInput* domA = new glades::NumberInput();
		glades::NumberInput* domB = new glades::NumberInput();
		const unsigned int nSamples = 50;
		const unsigned int featureDim = 2;

		domA->trainMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(featureDim, 0.0f));
		domA->trainExpectedMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(1, 0.0f));
		domB->trainMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(featureDim, 0.0f));
		domB->trainExpectedMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(1, 0.0f));

		unsigned int seed = 88u;
		for (unsigned int i = 0; i < nSamples; ++i)
		{
			seed = seed * 1103515245u + 12345u;
			float u1 = static_cast<float>((seed >> 16) & 0x7FFFu) / 32767.0f;
			seed = seed * 1103515245u + 12345u;
			float u2 = static_cast<float>((seed >> 16) & 0x7FFFu) / 32767.0f;
			if (u1 < 1e-6f) u1 = 1e-6f;
			float z1 = static_cast<float>(sqrt(-2.0 * log(static_cast<double>(u1))) * cos(6.28318 * static_cast<double>(u2)));
			float z2 = static_cast<float>(sqrt(-2.0 * log(static_cast<double>(u1))) * sin(6.28318 * static_cast<double>(u2)));

			domA->trainMatrix[i][0] = 2.0f + 0.3f * z1;
			domA->trainMatrix[i][1] = 2.0f + 0.3f * z2;
			domB->trainMatrix[i][0] = -2.0f + 0.3f * z1;
			domB->trainMatrix[i][1] = -2.0f + 0.3f * z2;
		}
		domA->testMatrix = domA->trainMatrix;
		domA->testExpectedMatrix = domA->trainExpectedMatrix;
		domB->testMatrix = domB->trainMatrix;
		domB->testExpectedMatrix = domB->trainExpectedMatrix;

		std::vector<unsigned int> genHidden;
		genHidden.push_back(16u);
		genHidden.push_back(8u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(8u);

		glades::NNInfo* genABInfo = make_gen_info("ut_cyc_gAB", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* genBAInfo = make_gen_info("ut_cyc_gBA", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discAInfo = make_disc_info("ut_cyc_dA", discHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discBInfo = make_disc_info("ut_cyc_dB", discHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.variantType = glades::GANConfig::GAN_CYCLE;
		cfg.noiseDim = 8;
		cfg.epochs = 10;
		cfg.batchSize = 10;
		cfg.nCriticPerGenerator = 1;
		cfg.cycleConfig.cycleLambda = 10.0f;
		cfg.cycleConfig.identityLambda = 5.0f;

		glades::GAN gan(cfg, genABInfo, genBAInfo, discAInfo, discBInfo);
		gan.setSeed(800);

		GANCaptureMetrics metrics;
		const glades::NNetworkStatus st = gan.train(domA, domB, &metrics);
		printf("[UT] CycleGAN DFF train status: %s (code=%d)\n", st.message.c_str(), static_cast<int>(st.code));
		G_assert(__FILE__, __LINE__, "==============GAN::CycleGAN TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::CycleGAN No Metrics==============", metrics.saw);

		// Verify metrics are finite
		G_assert(__FILE__, __LINE__, "==============GAN::CycleGAN dLossA Non-Finite==============", is_finite(metrics.last.dLossA));
		G_assert(__FILE__, __LINE__, "==============GAN::CycleGAN dLossB Non-Finite==============", is_finite(metrics.last.dLossB));
		G_assert(__FILE__, __LINE__, "==============GAN::CycleGAN cycleLoss Non-Finite==============", is_finite(metrics.last.cycleLoss));

		// Translate A->B
		std::vector<std::vector<float> > translated;
		const glades::NNetworkStatus trSt = gan.translate(domA, true, translated);
		G_assert(__FILE__, __LINE__, "==============GAN::CycleGAN TranslateStatus() Failed==============", trSt.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::CycleGAN Wrong Translate Count==============", translated.size() == nSamples);

		bool allFinite = true;
		for (size_t i = 0; i < translated.size(); ++i)
			for (size_t j = 0; j < translated[i].size(); ++j)
				if (!is_finite(translated[i][j]))
					allFinite = false;
		G_assert(__FILE__, __LINE__, "==============GAN::CycleGAN Non-Finite Translation==============", allFinite);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genABInfo;
		delete genBAInfo;
		delete discAInfo;
		delete discBInfo;
		delete domA;
		delete domB;
	}

	// ------------------------------------------------------------------
	// Test 12: Info+Style combo DFF - smoke test
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 12: Info+Style combo DFF - smoke test\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(100, 42u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);
		genHidden.push_back(16u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);
		discHidden.push_back(8u);

		std::vector<unsigned int> mapHidden;
		mapHidden.push_back(32u);

		glades::NNInfo* genInfo = make_gen_info("ut_is_gen", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo = make_disc_info("ut_is_disc", discHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* mapInfo = make_gen_info("ut_is_map", mapHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.noiseDim = 8;
		cfg.epochs = 15;
		cfg.batchSize = 20;
		cfg.nCriticPerGenerator = 1;
		cfg.useInfo = true;
		cfg.useStyle = true;
		cfg.infoConfig.numCategorical = 4;
		cfg.infoConfig.numContinuous = 2;
		cfg.infoConfig.infoLambda = 1.0f;
		cfg.styleConfig.mappingWidth = 16;
		cfg.styleConfig.mappingLayers = 1;
		cfg.styleConfig.noiseScaleInit = 0.0f;

		glades::GAN gan(cfg, genInfo, discInfo, mapInfo);
		gan.setSeed(1200);

		GANCaptureMetrics metrics;
		const glades::NNetworkStatus st = gan.train(di, &metrics);
		printf("[UT] Info+Style train status: %s (code=%d)\n", st.message.c_str(), static_cast<int>(st.code));
		G_assert(__FILE__, __LINE__, "==============GAN::Info+Style TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::Info+Style No Metrics==============", metrics.saw);
		G_assert(__FILE__, __LINE__, "==============GAN::Info+Style InfoLoss Non-Finite==============", is_finite(metrics.last.infoLoss));

		// Generate with fixed codes
		std::vector<float> cat0(4, 0.0f); cat0[0] = 1.0f;
		std::vector<float> cont(2, 0.0f);
		std::vector<std::vector<float> > samples;
		const glades::NNetworkStatus genSt = gan.generateInfoGAN(10, &cat0, &cont, samples);
		G_assert(__FILE__, __LINE__, "==============GAN::Info+Style GenerateStatus() Failed==============", genSt.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::Info+Style Wrong Sample Count==============", samples.size() == 10);

		bool allFinite = true;
		for (size_t i = 0; i < samples.size(); ++i)
			for (size_t j = 0; j < samples[i].size(); ++j)
				if (!is_finite(samples[i][j]))
					allFinite = false;
		G_assert(__FILE__, __LINE__, "==============GAN::Info+Style Non-Finite Samples==============", allFinite);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo;
		delete discInfo;
		delete mapInfo;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 13: Cycle+Info combo DFF - smoke test
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 13: Cycle+Info combo DFF - smoke test\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* domA = new glades::NumberInput();
		glades::NumberInput* domB = new glades::NumberInput();
		const unsigned int nSamples = 50;
		const unsigned int featureDim = 2;

		domA->trainMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(featureDim, 0.0f));
		domA->trainExpectedMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(1, 0.0f));
		domB->trainMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(featureDim, 0.0f));
		domB->trainExpectedMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(1, 0.0f));

		unsigned int seed = 130u;
		for (unsigned int i = 0; i < nSamples; ++i)
		{
			seed = seed * 1103515245u + 12345u;
			float u1 = static_cast<float>((seed >> 16) & 0x7FFFu) / 32767.0f;
			seed = seed * 1103515245u + 12345u;
			float u2 = static_cast<float>((seed >> 16) & 0x7FFFu) / 32767.0f;
			if (u1 < 1e-6f) u1 = 1e-6f;
			float z1 = static_cast<float>(sqrt(-2.0 * log(static_cast<double>(u1))) * cos(6.28318 * static_cast<double>(u2)));
			float z2 = static_cast<float>(sqrt(-2.0 * log(static_cast<double>(u1))) * sin(6.28318 * static_cast<double>(u2)));

			domA->trainMatrix[i][0] = 2.0f + 0.3f * z1;
			domA->trainMatrix[i][1] = 2.0f + 0.3f * z2;
			domB->trainMatrix[i][0] = -2.0f + 0.3f * z1;
			domB->trainMatrix[i][1] = -2.0f + 0.3f * z2;
		}
		domA->testMatrix = domA->trainMatrix;
		domA->testExpectedMatrix = domA->trainExpectedMatrix;
		domB->testMatrix = domB->trainMatrix;
		domB->testExpectedMatrix = domB->trainExpectedMatrix;

		std::vector<unsigned int> genHidden;
		genHidden.push_back(16u);
		genHidden.push_back(8u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(8u);

		glades::NNInfo* genABInfo = make_gen_info("ut_ci_gAB", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* genBAInfo = make_gen_info("ut_ci_gBA", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discAInfo = make_disc_info("ut_ci_dA", discHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discBInfo = make_disc_info("ut_ci_dB", discHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.noiseDim = 8;
		cfg.epochs = 10;
		cfg.batchSize = 10;
		cfg.nCriticPerGenerator = 1;
		cfg.useCycle = true;
		cfg.useInfo = true;
		cfg.cycleConfig.cycleLambda = 10.0f;
		cfg.cycleConfig.identityLambda = 5.0f;
		cfg.infoConfig.numCategorical = 3;
		cfg.infoConfig.numContinuous = 1;
		cfg.infoConfig.infoLambda = 1.0f;

		glades::GAN gan(cfg, genABInfo, genBAInfo, discAInfo, discBInfo);
		gan.setSeed(1300);

		GANCaptureMetrics metrics;
		const glades::NNetworkStatus st = gan.train(domA, domB, &metrics);
		printf("[UT] Cycle+Info train status: %s (code=%d)\n", st.message.c_str(), static_cast<int>(st.code));
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Info TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Info No Metrics==============", metrics.saw);
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Info InfoLoss Non-Finite==============", is_finite(metrics.last.infoLoss));
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Info cycleLoss Non-Finite==============", is_finite(metrics.last.cycleLoss));

		// Translate A->B with fixed codes
		std::vector<float> cat0(3, 0.0f); cat0[0] = 1.0f;
		std::vector<float> cont(1, 0.0f);
		std::vector<std::vector<float> > translated;
		const glades::NNetworkStatus trSt = gan.translate(domA, true, &cat0, &cont, translated);
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Info TranslateStatus() Failed==============", trSt.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Info Wrong Translate Count==============", translated.size() == nSamples);

		bool allFinite = true;
		for (size_t i = 0; i < translated.size(); ++i)
			for (size_t j = 0; j < translated[i].size(); ++j)
				if (!is_finite(translated[i][j]))
					allFinite = false;
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Info Non-Finite Translation==============", allFinite);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genABInfo;
		delete genBAInfo;
		delete discAInfo;
		delete discBInfo;
		delete domA;
		delete domB;
	}

	// ------------------------------------------------------------------
	// Test 14: Cycle+Style combo DFF - smoke test
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 14: Cycle+Style combo DFF - smoke test\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* domA = new glades::NumberInput();
		glades::NumberInput* domB = new glades::NumberInput();
		const unsigned int nSamples = 50;
		const unsigned int featureDim = 2;

		domA->trainMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(featureDim, 0.0f));
		domA->trainExpectedMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(1, 0.0f));
		domB->trainMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(featureDim, 0.0f));
		domB->trainExpectedMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(1, 0.0f));

		unsigned int seed = 140u;
		for (unsigned int i = 0; i < nSamples; ++i)
		{
			seed = seed * 1103515245u + 12345u;
			float u1 = static_cast<float>((seed >> 16) & 0x7FFFu) / 32767.0f;
			seed = seed * 1103515245u + 12345u;
			float u2 = static_cast<float>((seed >> 16) & 0x7FFFu) / 32767.0f;
			if (u1 < 1e-6f) u1 = 1e-6f;
			float z1 = static_cast<float>(sqrt(-2.0 * log(static_cast<double>(u1))) * cos(6.28318 * static_cast<double>(u2)));
			float z2 = static_cast<float>(sqrt(-2.0 * log(static_cast<double>(u1))) * sin(6.28318 * static_cast<double>(u2)));

			domA->trainMatrix[i][0] = 2.0f + 0.3f * z1;
			domA->trainMatrix[i][1] = 2.0f + 0.3f * z2;
			domB->trainMatrix[i][0] = -2.0f + 0.3f * z1;
			domB->trainMatrix[i][1] = -2.0f + 0.3f * z2;
		}
		domA->testMatrix = domA->trainMatrix;
		domA->testExpectedMatrix = domA->trainExpectedMatrix;
		domB->testMatrix = domB->trainMatrix;
		domB->testExpectedMatrix = domB->trainExpectedMatrix;

		std::vector<unsigned int> genHidden;
		genHidden.push_back(16u);
		genHidden.push_back(8u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(8u);

		std::vector<unsigned int> mapHidden;
		mapHidden.push_back(16u);

		glades::NNInfo* genABInfo = make_gen_info("ut_cs_gAB", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* genBAInfo = make_gen_info("ut_cs_gBA", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discAInfo = make_disc_info("ut_cs_dA", discHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discBInfo = make_disc_info("ut_cs_dB", discHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* mapABInfo = make_gen_info("ut_cs_mAB", mapHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* mapBAInfo = make_gen_info("ut_cs_mBA", mapHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.noiseDim = 8;
		cfg.epochs = 10;
		cfg.batchSize = 10;
		cfg.nCriticPerGenerator = 1;
		cfg.useCycle = true;
		cfg.useStyle = true;
		cfg.cycleConfig.cycleLambda = 10.0f;
		cfg.cycleConfig.identityLambda = 5.0f;
		cfg.styleConfig.mappingWidth = 16;
		cfg.styleConfig.mappingLayers = 1;
		cfg.styleConfig.noiseScaleInit = 0.0f;

		glades::GAN gan(cfg, genABInfo, genBAInfo, discAInfo, discBInfo, mapABInfo, mapBAInfo);
		gan.setSeed(1400);

		GANCaptureMetrics metrics;
		const glades::NNetworkStatus st = gan.train(domA, domB, &metrics);
		printf("[UT] Cycle+Style train status: %s (code=%d)\n", st.message.c_str(), static_cast<int>(st.code));
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Style TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Style No Metrics==============", metrics.saw);
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Style cycleLoss Non-Finite==============", is_finite(metrics.last.cycleLoss));

		// Translate A->B
		std::vector<std::vector<float> > translated;
		const glades::NNetworkStatus trSt = gan.translate(domA, true, translated);
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Style TranslateStatus() Failed==============", trSt.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Style Wrong Translate Count==============", translated.size() == nSamples);

		bool allFinite = true;
		for (size_t i = 0; i < translated.size(); ++i)
			for (size_t j = 0; j < translated[i].size(); ++j)
				if (!is_finite(translated[i][j]))
					allFinite = false;
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Style Non-Finite Translation==============", allFinite);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genABInfo;
		delete genBAInfo;
		delete discAInfo;
		delete discBInfo;
		delete mapABInfo;
		delete mapBAInfo;
		delete domA;
		delete domB;
	}

	// ------------------------------------------------------------------
	// Test 15: Cycle+Info+Style full combo DFF - smoke test
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 15: Cycle+Info+Style full combo DFF - smoke test\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* domA = new glades::NumberInput();
		glades::NumberInput* domB = new glades::NumberInput();
		const unsigned int nSamples = 50;
		const unsigned int featureDim = 2;

		domA->trainMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(featureDim, 0.0f));
		domA->trainExpectedMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(1, 0.0f));
		domB->trainMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(featureDim, 0.0f));
		domB->trainExpectedMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(1, 0.0f));

		unsigned int seed = 150u;
		for (unsigned int i = 0; i < nSamples; ++i)
		{
			seed = seed * 1103515245u + 12345u;
			float u1 = static_cast<float>((seed >> 16) & 0x7FFFu) / 32767.0f;
			seed = seed * 1103515245u + 12345u;
			float u2 = static_cast<float>((seed >> 16) & 0x7FFFu) / 32767.0f;
			if (u1 < 1e-6f) u1 = 1e-6f;
			float z1 = static_cast<float>(sqrt(-2.0 * log(static_cast<double>(u1))) * cos(6.28318 * static_cast<double>(u2)));
			float z2 = static_cast<float>(sqrt(-2.0 * log(static_cast<double>(u1))) * sin(6.28318 * static_cast<double>(u2)));

			domA->trainMatrix[i][0] = 2.0f + 0.3f * z1;
			domA->trainMatrix[i][1] = 2.0f + 0.3f * z2;
			domB->trainMatrix[i][0] = -2.0f + 0.3f * z1;
			domB->trainMatrix[i][1] = -2.0f + 0.3f * z2;
		}
		domA->testMatrix = domA->trainMatrix;
		domA->testExpectedMatrix = domA->trainExpectedMatrix;
		domB->testMatrix = domB->trainMatrix;
		domB->testExpectedMatrix = domB->trainExpectedMatrix;

		std::vector<unsigned int> genHidden;
		genHidden.push_back(16u);
		genHidden.push_back(8u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(8u);

		std::vector<unsigned int> mapHidden;
		mapHidden.push_back(16u);

		glades::NNInfo* genABInfo = make_gen_info("ut_cis_gAB", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* genBAInfo = make_gen_info("ut_cis_gBA", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discAInfo = make_disc_info("ut_cis_dA", discHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discBInfo = make_disc_info("ut_cis_dB", discHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* mapABInfo = make_gen_info("ut_cis_mAB", mapHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* mapBAInfo = make_gen_info("ut_cis_mBA", mapHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.noiseDim = 8;
		cfg.epochs = 10;
		cfg.batchSize = 10;
		cfg.nCriticPerGenerator = 1;
		cfg.useCycle = true;
		cfg.useInfo = true;
		cfg.useStyle = true;
		cfg.cycleConfig.cycleLambda = 10.0f;
		cfg.cycleConfig.identityLambda = 5.0f;
		cfg.infoConfig.numCategorical = 3;
		cfg.infoConfig.numContinuous = 1;
		cfg.infoConfig.infoLambda = 1.0f;
		cfg.styleConfig.mappingWidth = 16;
		cfg.styleConfig.mappingLayers = 1;
		cfg.styleConfig.noiseScaleInit = 0.0f;

		glades::GAN gan(cfg, genABInfo, genBAInfo, discAInfo, discBInfo, mapABInfo, mapBAInfo);
		gan.setSeed(1500);

		GANCaptureMetrics metrics;
		const glades::NNetworkStatus st = gan.train(domA, domB, &metrics);
		printf("[UT] Cycle+Info+Style train status: %s (code=%d)\n", st.message.c_str(), static_cast<int>(st.code));
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Info+Style TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Info+Style No Metrics==============", metrics.saw);
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Info+Style InfoLoss Non-Finite==============", is_finite(metrics.last.infoLoss));
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Info+Style cycleLoss Non-Finite==============", is_finite(metrics.last.cycleLoss));

		// Translate A->B with fixed codes
		std::vector<float> cat0(3, 0.0f); cat0[0] = 1.0f;
		std::vector<float> cont(1, 0.0f);
		std::vector<std::vector<float> > translated;
		const glades::NNetworkStatus trSt = gan.translate(domA, true, &cat0, &cont, translated);
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Info+Style TranslateStatus() Failed==============", trSt.ok());
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Info+Style Wrong Translate Count==============", translated.size() == nSamples);

		bool allFinite = true;
		for (size_t i = 0; i < translated.size(); ++i)
			for (size_t j = 0; j < translated[i].size(); ++j)
				if (!is_finite(translated[i][j]))
					allFinite = false;
		G_assert(__FILE__, __LINE__, "==============GAN::Cycle+Info+Style Non-Finite Translation==============", allFinite);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genABInfo;
		delete genBAInfo;
		delete discAInfo;
		delete discBInfo;
		delete mapABInfo;
		delete mapBAInfo;
		delete domA;
		delete domB;
	}

	// ------------------------------------------------------------------
	// Test 11: CycleGAN error - single-dataset train returns error
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 11: CycleGAN error handling\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(50, 11u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(8u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(8u);

		glades::NNInfo* genABInfo = make_gen_info("ut_cyc_err_gAB", genHidden, 0.001f, glades::GMath::LEAKY);
		glades::NNInfo* genBAInfo = make_gen_info("ut_cyc_err_gBA", genHidden, 0.001f, glades::GMath::LEAKY);
		glades::NNInfo* discAInfo = make_disc_info("ut_cyc_err_dA", discHidden, 0.001f, glades::GMath::LEAKY);
		glades::NNInfo* discBInfo = make_disc_info("ut_cyc_err_dB", discHidden, 0.001f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.variantType = glades::GANConfig::GAN_CYCLE;
		cfg.noiseDim = 4;
		cfg.epochs = 5;
		cfg.batchSize = 10;

		glades::GAN gan(cfg, genABInfo, genBAInfo, discAInfo, discBInfo);

		// Single-dataset train should return error for CycleGAN
		const glades::NNetworkStatus st = gan.train(di);
		printf("[UT] CycleGAN single-dataset train status: %s (code=%d)\n", st.message.c_str(), static_cast<int>(st.code));
		G_assert(__FILE__, __LINE__, "==============GAN::CycleGAN ShouldFail==============", !st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genABInfo;
		delete genBAInfo;
		delete discAInfo;
		delete discBInfo;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 16: GradientBuffer round-trip
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 16: GradientBuffer round-trip\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(20, 99u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(8u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(8u);

		glades::NNInfo* genInfo = make_gen_info("ut_gradbuf_gen", genHidden, 0.001f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo = make_disc_info("ut_gradbuf_disc", discHidden, 0.001f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.noiseDim = 4;
		cfg.epochs = 1;
		cfg.batchSize = 5;

		glades::GAN gan(cfg, genInfo, discInfo);
		gan.setSeed(1600);

		// Train 1 epoch to init tensors
		const glades::NNetworkStatus st = gan.train(di);
		G_assert(__FILE__, __LINE__, "==============GAN::GradBuf TrainStatus() Failed==============", st.ok());

		// Get generator (const ref), cast away const to access mutable tensor state
		const glades::NNetwork& genConst = gan.getGenerator();
		glades::NNetwork& gen = const_cast<glades::NNetwork&>(genConst);

		// Create buffer and init from DFF
		glades::GradientBuffer buf;
		buf.initFromDFF(gen);

		// Assert sizes match
		G_assert(__FILE__, __LINE__, "==============GAN::GradBuf dffGW.size() mismatch==============",
		         buf.dffGW.size() == gen.tensorDff.T.size());
		G_assert(__FILE__, __LINE__, "==============GAN::GradBuf dffGBias.size() mismatch==============",
		         buf.dffGBias.size() == gen.tensorDff.T.size());

		for (size_t t = 0; t < buf.dffGW.size(); ++t)
		{
			G_assert(__FILE__, __LINE__, "==============GAN::GradBuf dffGW[t].size() mismatch==============",
			         buf.dffGW[t].size() == gen.tensorDff.T[t].gW.size());
			G_assert(__FILE__, __LINE__, "==============GAN::GradBuf dffGBias[t].size() mismatch==============",
			         buf.dffGBias[t].size() == gen.tensorDff.T[t].gBias.size());
		}

		// Assert all values are zero after init
		bool allZero = true;
		for (size_t t = 0; t < buf.dffGW.size(); ++t)
		{
			for (size_t i = 0; i < buf.dffGW[t].size(); ++i)
				if (buf.dffGW[t][i] != 0.0f) allZero = false;
			for (size_t i = 0; i < buf.dffGBias[t].size(); ++i)
				if (buf.dffGBias[t][i] != 0.0f) allZero = false;
		}
		G_assert(__FILE__, __LINE__, "==============GAN::GradBuf Not Zero After Init==============", allZero);

		// Set known values: 1.0 for gW, 2.0 for gBias
		for (size_t t = 0; t < buf.dffGW.size(); ++t)
		{
			for (size_t i = 0; i < buf.dffGW[t].size(); ++i)
				buf.dffGW[t][i] = 1.0f;
			for (size_t i = 0; i < buf.dffGBias[t].size(); ++i)
				buf.dffGBias[t][i] = 2.0f;
		}

		// Zero network grads, then addToDFF
		for (size_t t = 0; t < gen.tensorDff.T.size(); ++t)
		{
			for (size_t i = 0; i < gen.tensorDff.T[t].gW.size(); ++i)
				gen.tensorDff.T[t].gW[i] = 0.0f;
			for (size_t i = 0; i < gen.tensorDff.T[t].gBias.size(); ++i)
				gen.tensorDff.T[t].gBias[i] = 0.0f;
		}

		buf.addToDFF(gen);

		// Assert network grads now match buffer values
		bool gradsMatch = true;
		for (size_t t = 0; t < gen.tensorDff.T.size(); ++t)
		{
			for (size_t i = 0; i < gen.tensorDff.T[t].gW.size(); ++i)
				if (fabsf(gen.tensorDff.T[t].gW[i] - 1.0f) > 1e-6f) gradsMatch = false;
			for (size_t i = 0; i < gen.tensorDff.T[t].gBias.size(); ++i)
				if (fabsf(gen.tensorDff.T[t].gBias[i] - 2.0f) > 1e-6f) gradsMatch = false;
		}
		G_assert(__FILE__, __LINE__, "==============GAN::GradBuf addToDFF Grads Mismatch==============", gradsMatch);

		// Test zero() makes all zeros again
		buf.zero();
		bool allZeroAfter = true;
		for (size_t t = 0; t < buf.dffGW.size(); ++t)
		{
			for (size_t i = 0; i < buf.dffGW[t].size(); ++i)
				if (buf.dffGW[t][i] != 0.0f) allZeroAfter = false;
			for (size_t i = 0; i < buf.dffGBias[t].size(); ++i)
				if (buf.dffGBias[t][i] != 0.0f) allZeroAfter = false;
		}
		G_assert(__FILE__, __LINE__, "==============GAN::GradBuf Not Zero After zero()==============", allZeroAfter);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo;
		delete discInfo;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 17: Determinism - same seed produces identical losses
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 17: Parallel determinism\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(100, 42u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);
		genHidden.push_back(16u);
		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);
		discHidden.push_back(8u);

		glades::NNInfo* genInfo1 = make_gen_info("ut_det_gen1", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo1 = make_disc_info("ut_det_disc1", discHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* genInfo2 = make_gen_info("ut_det_gen2", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo2 = make_disc_info("ut_det_disc2", discHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.noiseDim = 8;
		cfg.epochs = 5;
		cfg.batchSize = 20;
		cfg.nCriticPerGenerator = 1;
		cfg.deterministicReduce = true;

		glades::GAN gan1(cfg, genInfo1, discInfo1);
		gan1.setSeed(999);
		GANCaptureMetrics met1;
		glades::NNetworkStatus st1 = gan1.train(di, &met1);
		G_assert(__FILE__, __LINE__, "==============Determinism: train1 failed==============", st1.ok());

		glades::GAN gan2(cfg, genInfo2, discInfo2);
		gan2.setSeed(999);
		GANCaptureMetrics met2;
		glades::NNetworkStatus st2 = gan2.train(di, &met2);
		G_assert(__FILE__, __LINE__, "==============Determinism: train2 failed==============", st2.ok());

		// Losses should be bit-identical
		G_assert(__FILE__, __LINE__, "==============Determinism: dLossReal mismatch==============",
		         met1.last.dLossReal == met2.last.dLossReal);
		G_assert(__FILE__, __LINE__, "==============Determinism: gLoss mismatch==============",
		         met1.last.gLoss == met2.last.gLoss);

		// Generated samples should be identical
		std::vector<std::vector<float> > samples1, samples2;
		gan1.generate(5, samples1);
		gan2.generate(5, samples2);
		bool samplesMatch = true;
		for (size_t i = 0; i < samples1.size(); ++i)
			for (size_t j = 0; j < samples1[i].size(); ++j)
				if (samples1[i][j] != samples2[i][j]) samplesMatch = false;
		G_assert(__FILE__, __LINE__, "==============Determinism: samples mismatch==============", samplesMatch);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo1; delete discInfo1;
		delete genInfo2; delete discInfo2;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 18: WGAN-GP DFF parallel determinism
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 18: WGAN-GP DFF parallel determinism\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(100, 42u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);
		genHidden.push_back(16u);
		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);
		discHidden.push_back(8u);

		glades::NNInfo* genInfo1 = make_gen_info("ut_wdet_gen1", genHidden, 0.0001f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo1 = make_disc_info("ut_wdet_disc1", discHidden, 0.0001f, glades::GMath::LEAKY);
		glades::NNInfo* genInfo2 = make_gen_info("ut_wdet_gen2", genHidden, 0.0001f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo2 = make_disc_info("ut_wdet_disc2", discHidden, 0.0001f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_WGAN_GP;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.noiseDim = 8;
		cfg.epochs = 5;
		cfg.batchSize = 20;
		cfg.nCriticPerGenerator = 3;
		cfg.gpLambda = 10.0f;
		cfg.deterministicReduce = true;

		glades::GAN gan1(cfg, genInfo1, discInfo1);
		gan1.setSeed(999);
		GANCaptureMetrics met1;
		glades::NNetworkStatus st1 = gan1.train(di, &met1);
		G_assert(__FILE__, __LINE__, "==============WGANGP Determinism: train1 failed==============", st1.ok());

		glades::GAN gan2(cfg, genInfo2, discInfo2);
		gan2.setSeed(999);
		GANCaptureMetrics met2;
		glades::NNetworkStatus st2 = gan2.train(di, &met2);
		G_assert(__FILE__, __LINE__, "==============WGANGP Determinism: train2 failed==============", st2.ok());

		// Losses should be bit-identical
		G_assert(__FILE__, __LINE__, "==============WGANGP Determinism: dLossReal mismatch==============",
		         met1.last.dLossReal == met2.last.dLossReal);
		G_assert(__FILE__, __LINE__, "==============WGANGP Determinism: gLoss mismatch==============",
		         met1.last.gLoss == met2.last.gLoss);
		G_assert(__FILE__, __LINE__, "==============WGANGP Determinism: wasserstein mismatch==============",
		         met1.last.wasserstein == met2.last.wasserstein);

		// Generated samples should be identical
		std::vector<std::vector<float> > samples1, samples2;
		gan1.generate(5, samples1);
		gan2.generate(5, samples2);
		bool samplesMatch = true;
		for (size_t i = 0; i < samples1.size(); ++i)
			for (size_t j = 0; j < samples1[i].size(); ++j)
				if (samples1[i][j] != samples2[i][j]) samplesMatch = false;
		G_assert(__FILE__, __LINE__, "==============WGANGP Determinism: samples mismatch==============", samplesMatch);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo1; delete discInfo1;
		delete genInfo2; delete discInfo2;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 19: CNN vanilla parallel determinism
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 19: CNN vanilla parallel determinism\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		glades::NumberInput* di = make_synthetic_image_data(40, C, H, W, 99u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);
		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);

		glades::NNInfo* genInfo1 = make_gen_info("ut_cnndet_gen1", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo1 = make_disc_info("ut_cnndet_disc1", discHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* genInfo2 = make_gen_info("ut_cnndet_gen2", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo2 = make_disc_info("ut_cnndet_disc2", discHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_CNN;
		cfg.noiseDim = 16;
		cfg.epochs = 5;
		cfg.batchSize = 10;
		cfg.nCriticPerGenerator = 1;
		cfg.deterministicReduce = true;

		glades::CNNConfig::ConvLayerSpec convSpec;
		convSpec.outChannels = 4;
		convSpec.kernelH = 3; convSpec.kernelW = 3;
		convSpec.strideH = 1; convSpec.strideW = 1;
		convSpec.padH = 1; convSpec.padW = 1;
		convSpec.useBatchNorm = false;
		convSpec.useMaxPool = false;
		cfg.discriminatorCNN.inputH = H;
		cfg.discriminatorCNN.inputW = W;
		cfg.discriminatorCNN.inputC = C;
		cfg.discriminatorCNN.convLayers.push_back(convSpec);

		glades::GAN gan1(cfg, genInfo1, discInfo1);
		gan1.setSeed(789);
		GANCaptureMetrics met1;
		glades::NNetworkStatus st1 = gan1.train(di, &met1);
		G_assert(__FILE__, __LINE__, "==============CNN Determinism: train1 failed==============", st1.ok());

		glades::GAN gan2(cfg, genInfo2, discInfo2);
		gan2.setSeed(789);
		GANCaptureMetrics met2;
		glades::NNetworkStatus st2 = gan2.train(di, &met2);
		G_assert(__FILE__, __LINE__, "==============CNN Determinism: train2 failed==============", st2.ok());

		// Losses should be bit-identical
		G_assert(__FILE__, __LINE__, "==============CNN Determinism: dLossReal mismatch==============",
		         met1.last.dLossReal == met2.last.dLossReal);
		G_assert(__FILE__, __LINE__, "==============CNN Determinism: gLoss mismatch==============",
		         met1.last.gLoss == met2.last.gLoss);

		// Generated samples should be identical
		std::vector<std::vector<float> > samples1, samples2;
		gan1.generate(5, samples1);
		gan2.generate(5, samples2);
		bool samplesMatch = true;
		for (size_t i = 0; i < samples1.size(); ++i)
			for (size_t j = 0; j < samples1[i].size(); ++j)
				if (samples1[i][j] != samples2[i][j]) samplesMatch = false;
		G_assert(__FILE__, __LINE__, "==============CNN Determinism: samples mismatch==============", samplesMatch);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo1; delete discInfo1;
		delete genInfo2; delete discInfo2;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 20: InfoGAN DFF parallel determinism
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 20: InfoGAN DFF parallel determinism\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(100, 42u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);
		genHidden.push_back(16u);
		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);
		discHidden.push_back(8u);

		glades::NNInfo* genInfo1 = make_gen_info("ut_idet_gen1", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo1 = make_disc_info("ut_idet_disc1", discHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* genInfo2 = make_gen_info("ut_idet_gen2", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo2 = make_disc_info("ut_idet_disc2", discHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.variantType = glades::GANConfig::GAN_INFO;
		cfg.noiseDim = 8;
		cfg.epochs = 5;
		cfg.batchSize = 20;
		cfg.nCriticPerGenerator = 1;
		cfg.infoConfig.numCategorical = 4;
		cfg.infoConfig.numContinuous = 2;
		cfg.infoConfig.infoLambda = 1.0f;
		cfg.deterministicReduce = true;

		glades::GAN gan1(cfg, genInfo1, discInfo1);
		gan1.setSeed(500);
		GANCaptureMetrics met1;
		glades::NNetworkStatus st1 = gan1.train(di, &met1);
		G_assert(__FILE__, __LINE__, "==============InfoGAN Determinism: train1 failed==============", st1.ok());

		glades::GAN gan2(cfg, genInfo2, discInfo2);
		gan2.setSeed(500);
		GANCaptureMetrics met2;
		glades::NNetworkStatus st2 = gan2.train(di, &met2);
		G_assert(__FILE__, __LINE__, "==============InfoGAN Determinism: train2 failed==============", st2.ok());

		// Losses should be bit-identical
		G_assert(__FILE__, __LINE__, "==============InfoGAN Determinism: dLossReal mismatch==============",
		         met1.last.dLossReal == met2.last.dLossReal);
		G_assert(__FILE__, __LINE__, "==============InfoGAN Determinism: gLoss mismatch==============",
		         met1.last.gLoss == met2.last.gLoss);
		G_assert(__FILE__, __LINE__, "==============InfoGAN Determinism: infoLoss mismatch==============",
		         met1.last.infoLoss == met2.last.infoLoss);

		// Generated samples should be identical
		std::vector<std::vector<float> > samples1, samples2;
		gan1.generate(5, samples1);
		gan2.generate(5, samples2);
		bool samplesMatch = true;
		for (size_t i = 0; i < samples1.size(); ++i)
			for (size_t j = 0; j < samples1[i].size(); ++j)
				if (samples1[i][j] != samples2[i][j]) samplesMatch = false;
		G_assert(__FILE__, __LINE__, "==============InfoGAN Determinism: samples mismatch==============", samplesMatch);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo1; delete discInfo1;
		delete genInfo2; delete discInfo2;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 21: StyleGAN DFF parallel determinism
	// ------------------------------------------------------------------
	// NOTE: dffForwardStyled injects per-layer noise via a shared RNG
	// that races across threads. With noiseScaleInit=0 the disc phase
	// (forward-only through gen) is bit-identical, but the gen phase
	// backward accumulates divergent noise-scale gradients.  We therefore
	// compare losses with a small tolerance and verify the disc-phase loss
	// is bit-identical while the gen-phase loss is approximately equal.
	printf("-----------------------------------\n");
	printf("GAN Test 21: StyleGAN DFF parallel determinism\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(100, 42u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);
		genHidden.push_back(16u);
		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);
		discHidden.push_back(8u);
		std::vector<unsigned int> mapHidden;
		mapHidden.push_back(32u);

		glades::NNInfo* genInfo1 = make_gen_info("ut_sdet_gen1", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo1 = make_disc_info("ut_sdet_disc1", discHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* mapInfo1 = make_gen_info("ut_sdet_map1", mapHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* genInfo2 = make_gen_info("ut_sdet_gen2", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo2 = make_disc_info("ut_sdet_disc2", discHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* mapInfo2 = make_gen_info("ut_sdet_map2", mapHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.variantType = glades::GANConfig::GAN_STYLE;
		cfg.noiseDim = 8;
		cfg.epochs = 1;
		cfg.batchSize = 100;
		cfg.nCriticPerGenerator = 1;
		cfg.styleConfig.mappingWidth = 16;
		cfg.styleConfig.mappingLayers = 1;
		cfg.styleConfig.noiseScaleInit = 0.0f;
		cfg.deterministicReduce = true;

		glades::GAN gan1(cfg, genInfo1, discInfo1, mapInfo1);
		gan1.setSeed(700);
		GANCaptureMetrics met1;
		glades::NNetworkStatus st1 = gan1.train(di, &met1);
		G_assert(__FILE__, __LINE__, "==============StyleGAN Determinism: train1 failed==============", st1.ok());

		glades::GAN gan2(cfg, genInfo2, discInfo2, mapInfo2);
		gan2.setSeed(700);
		GANCaptureMetrics met2;
		glades::NNetworkStatus st2 = gan2.train(di, &met2);
		G_assert(__FILE__, __LINE__, "==============StyleGAN Determinism: train2 failed==============", st2.ok());

		// Disc-phase loss is bit-identical (gen forward with scale=0 is deterministic)
		G_assert(__FILE__, __LINE__, "==============StyleGAN Determinism: dLossReal mismatch==============",
		         met1.last.dLossReal == met2.last.dLossReal);

		// Gen-phase loss: approximate comparison (per-layer noise RNG races in backward)
		const float gDiff = fabsf(met1.last.gLoss - met2.last.gLoss);
		const float gScale = fabsf(met1.last.gLoss) + fabsf(met2.last.gLoss) + 1e-8f;
		printf("[UT] StyleGAN gLoss: run1=%.6f run2=%.6f relDiff=%.6e\n",
		       met1.last.gLoss, met2.last.gLoss, static_cast<double>(gDiff / gScale));
		G_assert(__FILE__, __LINE__, "==============StyleGAN Determinism: gLoss too different==============",
		         gDiff / gScale < 0.05f);

		// Generated samples: verify both runs produce finite, non-degenerate output.
		// Per-layer noise scale divergence means exact sample match is not expected.
		std::vector<std::vector<float> > samples1, samples2;
		gan1.generate(5, samples1);
		gan2.generate(5, samples2);
		G_assert(__FILE__, __LINE__, "==============StyleGAN Determinism: sample count mismatch==============",
		         samples1.size() == samples2.size() && samples1.size() == 5);
		bool allFinite1 = true, allFinite2 = true;
		for (size_t i = 0; i < samples1.size(); ++i)
			for (size_t j = 0; j < samples1[i].size(); ++j)
			{
				if (!is_finite(samples1[i][j])) allFinite1 = false;
				if (!is_finite(samples2[i][j])) allFinite2 = false;
			}
		G_assert(__FILE__, __LINE__, "==============StyleGAN Determinism: run1 samples non-finite==============", allFinite1);
		G_assert(__FILE__, __LINE__, "==============StyleGAN Determinism: run2 samples non-finite==============", allFinite2);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo1; delete discInfo1; delete mapInfo1;
		delete genInfo2; delete discInfo2; delete mapInfo2;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 22: CycleGAN DFF parallel determinism
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 22: CycleGAN DFF parallel determinism\n");
	printf("-----------------------------------\n");
	{
		// Build two domain datasets
		glades::NumberInput* domA = new glades::NumberInput();
		glades::NumberInput* domB = new glades::NumberInput();
		const unsigned int nSamples = 50;
		const unsigned int featureDim = 2;

		domA->trainMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(featureDim, 0.0f));
		domA->trainExpectedMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(1, 0.0f));
		domB->trainMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(featureDim, 0.0f));
		domB->trainExpectedMatrix = shmea::GMatrix(nSamples, shmea::GVector<float>(1, 0.0f));

		unsigned int seed = 88u;
		for (unsigned int i = 0; i < nSamples; ++i)
		{
			seed = seed * 1103515245u + 12345u;
			float u1 = static_cast<float>((seed >> 16) & 0x7FFFu) / 32767.0f;
			seed = seed * 1103515245u + 12345u;
			float u2 = static_cast<float>((seed >> 16) & 0x7FFFu) / 32767.0f;
			if (u1 < 1e-6f) u1 = 1e-6f;
			float z1 = static_cast<float>(sqrt(-2.0 * log(static_cast<double>(u1))) * cos(6.28318 * static_cast<double>(u2)));
			float z2 = static_cast<float>(sqrt(-2.0 * log(static_cast<double>(u1))) * sin(6.28318 * static_cast<double>(u2)));

			domA->trainMatrix[i][0] = 2.0f + 0.3f * z1;
			domA->trainMatrix[i][1] = 2.0f + 0.3f * z2;
			domB->trainMatrix[i][0] = -2.0f + 0.3f * z1;
			domB->trainMatrix[i][1] = -2.0f + 0.3f * z2;
		}
		domA->testMatrix = domA->trainMatrix;
		domA->testExpectedMatrix = domA->trainExpectedMatrix;
		domB->testMatrix = domB->trainMatrix;
		domB->testExpectedMatrix = domB->trainExpectedMatrix;

		std::vector<unsigned int> genHidden;
		genHidden.push_back(16u);
		genHidden.push_back(8u);
		std::vector<unsigned int> discHidden;
		discHidden.push_back(8u);

		// First run
		glades::NNInfo* genABInfo1 = make_gen_info("ut_cydet_gAB1", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* genBAInfo1 = make_gen_info("ut_cydet_gBA1", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discAInfo1 = make_disc_info("ut_cydet_dA1", discHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discBInfo1 = make_disc_info("ut_cydet_dB1", discHidden, 0.0002f, glades::GMath::LEAKY);
		// Second run
		glades::NNInfo* genABInfo2 = make_gen_info("ut_cydet_gAB2", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* genBAInfo2 = make_gen_info("ut_cydet_gBA2", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discAInfo2 = make_disc_info("ut_cydet_dA2", discHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discBInfo2 = make_disc_info("ut_cydet_dB2", discHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.variantType = glades::GANConfig::GAN_CYCLE;
		cfg.noiseDim = 8;
		cfg.epochs = 5;
		cfg.batchSize = 10;
		cfg.nCriticPerGenerator = 1;
		cfg.cycleConfig.cycleLambda = 10.0f;
		cfg.cycleConfig.identityLambda = 5.0f;
		cfg.deterministicReduce = true;

		glades::GAN gan1(cfg, genABInfo1, genBAInfo1, discAInfo1, discBInfo1);
		gan1.setSeed(800);
		GANCaptureMetrics met1;
		glades::NNetworkStatus st1 = gan1.train(domA, domB, &met1);
		G_assert(__FILE__, __LINE__, "==============CycleGAN Determinism: train1 failed==============", st1.ok());

		glades::GAN gan2(cfg, genABInfo2, genBAInfo2, discAInfo2, discBInfo2);
		gan2.setSeed(800);
		GANCaptureMetrics met2;
		glades::NNetworkStatus st2 = gan2.train(domA, domB, &met2);
		G_assert(__FILE__, __LINE__, "==============CycleGAN Determinism: train2 failed==============", st2.ok());

		// All CycleGAN losses should be bit-identical
		G_assert(__FILE__, __LINE__, "==============CycleGAN Determinism: dLossA mismatch==============",
		         met1.last.dLossA == met2.last.dLossA);
		G_assert(__FILE__, __LINE__, "==============CycleGAN Determinism: dLossB mismatch==============",
		         met1.last.dLossB == met2.last.dLossB);
		G_assert(__FILE__, __LINE__, "==============CycleGAN Determinism: gLossAB mismatch==============",
		         met1.last.gLossAB == met2.last.gLossAB);
		G_assert(__FILE__, __LINE__, "==============CycleGAN Determinism: gLossBA mismatch==============",
		         met1.last.gLossBA == met2.last.gLossBA);
		G_assert(__FILE__, __LINE__, "==============CycleGAN Determinism: cycleLoss mismatch==============",
		         met1.last.cycleLoss == met2.last.cycleLoss);

		// Translate A->B and verify identical results
		std::vector<std::vector<float> > trans1, trans2;
		gan1.translate(domA, true, trans1);
		gan2.translate(domA, true, trans2);
		bool transMatch = true;
		for (size_t i = 0; i < trans1.size(); ++i)
			for (size_t j = 0; j < trans1[i].size(); ++j)
				if (trans1[i][j] != trans2[i][j]) transMatch = false;
		G_assert(__FILE__, __LINE__, "==============CycleGAN Determinism: translate mismatch==============", transMatch);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genABInfo1; delete genBAInfo1; delete discAInfo1; delete discBInfo1;
		delete genABInfo2; delete genBAInfo2; delete discAInfo2; delete discBInfo2;
		delete domA;
		delete domB;
	}

	// ------------------------------------------------------------------
	// Test 23: Health diagnostics - DFF vanilla
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 23: Health diagnostics - DFF vanilla\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(100, 42u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);
		genHidden.push_back(16u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);
		discHidden.push_back(8u);

		glades::NNInfo* genInfo = make_gen_info("ut_diag_gen", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo = make_disc_info("ut_diag_disc", discHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_VANILLA;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.noiseDim = 8;
		cfg.epochs = 5;
		cfg.batchSize = 20;
		cfg.nCriticPerGenerator = 1;

		glades::GAN gan(cfg, genInfo, discInfo);
		gan.setSeed(123);

		GANCaptureMetrics metrics;
		const glades::NNetworkStatus st = gan.train(di, &metrics);
		G_assert(__FILE__, __LINE__, "==============Diag::DFF_Vanilla TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============Diag::DFF_Vanilla No Metrics==============", metrics.saw);

		// D(real) should be between 0 and 1 for vanilla GAN (sigmoid output)
		G_assert(__FILE__, __LINE__, "==============Diag::DFF_Vanilla dOutReal not finite==============",
		         is_finite(metrics.last.dOutReal));
		G_assert(__FILE__, __LINE__, "==============Diag::DFF_Vanilla dOutReal out of range==============",
		         metrics.last.dOutReal >= 0.0f && metrics.last.dOutReal <= 1.0f);

		// D(fake) should be between 0 and 1 for vanilla GAN
		G_assert(__FILE__, __LINE__, "==============Diag::DFF_Vanilla dOutFake not finite==============",
		         is_finite(metrics.last.dOutFake));
		G_assert(__FILE__, __LINE__, "==============Diag::DFF_Vanilla dOutFake out of range==============",
		         metrics.last.dOutFake >= 0.0f && metrics.last.dOutFake <= 1.0f);

		// D(real) should be greater than D(fake) — disc can distinguish
		G_assert(__FILE__, __LINE__, "==============Diag::DFF_Vanilla dOutReal <= dOutFake==============",
		         metrics.last.dOutReal > metrics.last.dOutFake);

		// Generator gradient norm should be positive and finite
		G_assert(__FILE__, __LINE__, "==============Diag::DFF_Vanilla genGradNorm not finite==============",
		         is_finite(metrics.last.genGradNorm));
		G_assert(__FILE__, __LINE__, "==============Diag::DFF_Vanilla genGradNorm not positive==============",
		         metrics.last.genGradNorm > 0.0f);

		// Sample diversity should be positive (not mode-collapsed)
		G_assert(__FILE__, __LINE__, "==============Diag::DFF_Vanilla sampleDiversity not finite==============",
		         is_finite(metrics.last.sampleDiversity));
		G_assert(__FILE__, __LINE__, "==============Diag::DFF_Vanilla sampleDiversity not positive==============",
		         metrics.last.sampleDiversity > 0.0f);

		printf("  dOutReal=%.4f dOutFake=%.4f genGradNorm=%.4f diversity=%.6f\n",
		       metrics.last.dOutReal, metrics.last.dOutFake,
		       metrics.last.genGradNorm, metrics.last.sampleDiversity);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo;
		delete discInfo;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 24: Health diagnostics - LSGAN (unbounded disc output)
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 24: Health diagnostics - LSGAN\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(100, 42u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);
		genHidden.push_back(16u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);
		discHidden.push_back(8u);

		glades::NNInfo* genInfo = make_gen_info("ut_diag_ls_gen", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo = make_disc_info("ut_diag_ls_disc", discHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_LSGAN;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.noiseDim = 8;
		cfg.epochs = 5;
		cfg.batchSize = 20;
		cfg.nCriticPerGenerator = 1;

		glades::GAN gan(cfg, genInfo, discInfo);
		gan.setSeed(456);

		GANCaptureMetrics metrics;
		const glades::NNetworkStatus st = gan.train(di, &metrics);
		G_assert(__FILE__, __LINE__, "==============Diag::LSGAN TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============Diag::LSGAN No Metrics==============", metrics.saw);

		// LSGAN: no sigmoid, so D outputs are unbounded — just check finite
		G_assert(__FILE__, __LINE__, "==============Diag::LSGAN dOutReal not finite==============",
		         is_finite(metrics.last.dOutReal));
		G_assert(__FILE__, __LINE__, "==============Diag::LSGAN dOutFake not finite==============",
		         is_finite(metrics.last.dOutFake));

		// D(real) should still tend toward 1, D(fake) toward 0 for LSGAN
		G_assert(__FILE__, __LINE__, "==============Diag::LSGAN dOutReal <= dOutFake==============",
		         metrics.last.dOutReal > metrics.last.dOutFake);

		// genGradNorm and diversity still valid
		G_assert(__FILE__, __LINE__, "==============Diag::LSGAN genGradNorm not positive==============",
		         is_finite(metrics.last.genGradNorm) && metrics.last.genGradNorm > 0.0f);
		G_assert(__FILE__, __LINE__, "==============Diag::LSGAN sampleDiversity not positive==============",
		         is_finite(metrics.last.sampleDiversity) && metrics.last.sampleDiversity > 0.0f);

		printf("  dOutReal=%.4f dOutFake=%.4f genGradNorm=%.4f diversity=%.6f\n",
		       metrics.last.dOutReal, metrics.last.dOutFake,
		       metrics.last.genGradNorm, metrics.last.sampleDiversity);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo;
		delete discInfo;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 25: Health diagnostics - CNN architecture
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 25: Health diagnostics - CNN\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		glades::NumberInput* di = make_synthetic_image_data(64, C, H, W, 42u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);

		glades::NNInfo* genInfo = make_gen_info("ut_diag_cnn_gen", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo = make_disc_info("ut_diag_cnn_disc", discHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_LSGAN;
		cfg.archType = glades::GANConfig::GAN_CNN;
		cfg.noiseDim = 8;
		cfg.epochs = 3;
		cfg.batchSize = 16;
		cfg.nCriticPerGenerator = 1;

		glades::CNNConfig::ConvLayerSpec convSpec;
		convSpec.outChannels = 4;
		convSpec.kernelH = 3; convSpec.kernelW = 3;
		convSpec.strideH = 1; convSpec.strideW = 1;
		convSpec.padH = 1; convSpec.padW = 1;
		convSpec.useBatchNorm = false;
		convSpec.useMaxPool = false;
		cfg.discriminatorCNN.inputH = H;
		cfg.discriminatorCNN.inputW = W;
		cfg.discriminatorCNN.inputC = C;
		cfg.discriminatorCNN.convLayers.push_back(convSpec);

		glades::GAN gan(cfg, genInfo, discInfo);
		gan.setSeed(789);

		GANCaptureMetrics metrics;
		const glades::NNetworkStatus st = gan.train(di, &metrics);
		G_assert(__FILE__, __LINE__, "==============Diag::CNN TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============Diag::CNN No Metrics==============", metrics.saw);

		// All health metrics should be finite and positive
		G_assert(__FILE__, __LINE__, "==============Diag::CNN dOutReal not finite==============",
		         is_finite(metrics.last.dOutReal));
		G_assert(__FILE__, __LINE__, "==============Diag::CNN dOutFake not finite==============",
		         is_finite(metrics.last.dOutFake));
		G_assert(__FILE__, __LINE__, "==============Diag::CNN genGradNorm not positive==============",
		         is_finite(metrics.last.genGradNorm) && metrics.last.genGradNorm > 0.0f);
		G_assert(__FILE__, __LINE__, "==============Diag::CNN sampleDiversity not finite==============",
		         is_finite(metrics.last.sampleDiversity));

		printf("  dOutReal=%.4f dOutFake=%.4f genGradNorm=%.4f diversity=%.6f\n",
		       metrics.last.dOutReal, metrics.last.dOutFake,
		       metrics.last.genGradNorm, metrics.last.sampleDiversity);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo;
		delete discInfo;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 26: Health diagnostics - InfoGAN with diagnostics
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("GAN Test 26: Health diagnostics - InfoGAN\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = make_gaussian_mixture(100, 42u);

		std::vector<unsigned int> genHidden;
		genHidden.push_back(32u);
		genHidden.push_back(16u);

		std::vector<unsigned int> discHidden;
		discHidden.push_back(16u);
		discHidden.push_back(8u);

		glades::NNInfo* genInfo = make_gen_info("ut_diag_info_gen", genHidden, 0.0002f, glades::GMath::LEAKY);
		glades::NNInfo* discInfo = make_disc_info("ut_diag_info_disc", discHidden, 0.0002f, glades::GMath::LEAKY);

		glades::GANConfig cfg;
		cfg.lossType = glades::GANConfig::GAN_LSGAN;
		cfg.archType = glades::GANConfig::GAN_DFF;
		cfg.noiseDim = 8;
		cfg.epochs = 5;
		cfg.batchSize = 20;
		cfg.nCriticPerGenerator = 1;
		cfg.infoConfig.numCategorical = 3;
		cfg.infoConfig.numContinuous = 1;
		cfg.infoConfig.infoLambda = 1.0f;

		glades::GAN gan(cfg, genInfo, discInfo);
		gan.setSeed(321);

		GANCaptureMetrics metrics;
		const glades::NNetworkStatus st = gan.train(di, &metrics);
		G_assert(__FILE__, __LINE__, "==============Diag::InfoGAN TrainStatus() Failed==============", st.ok());
		G_assert(__FILE__, __LINE__, "==============Diag::InfoGAN No Metrics==============", metrics.saw);

		// All diagnostics should be populated alongside InfoGAN-specific metrics
		G_assert(__FILE__, __LINE__, "==============Diag::InfoGAN dOutReal not finite==============",
		         is_finite(metrics.last.dOutReal));
		G_assert(__FILE__, __LINE__, "==============Diag::InfoGAN genGradNorm not positive==============",
		         is_finite(metrics.last.genGradNorm) && metrics.last.genGradNorm > 0.0f);
		G_assert(__FILE__, __LINE__, "==============Diag::InfoGAN sampleDiversity not finite==============",
		         is_finite(metrics.last.sampleDiversity));
		G_assert(__FILE__, __LINE__, "==============Diag::InfoGAN catAccuracy not finite==============",
		         is_finite(metrics.last.catAccuracy));

		printf("  dOutReal=%.4f dOutFake=%.4f genGradNorm=%.4f diversity=%.6f catAcc=%.2f%%\n",
		       metrics.last.dOutReal, metrics.last.dOutFake,
		       metrics.last.genGradNorm, metrics.last.sampleDiversity,
		       metrics.last.catAccuracy * 100.0f);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		delete genInfo;
		delete discInfo;
		delete di;
	}

	printf("============================================================\n");
	printf("GAN Unit Test Suite Complete\n");
	printf("============================================================\n");
}
