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

#include "cnn-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/training_callbacks.h"
#include "../../../Backend/Machine Learning/Networks/training_config.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/DataObjects/ImageInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
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

// Callback that captures epoch metrics and optionally stops early.
struct CaptureMetricsCb : public glades::ITrainingCallbacks
{
	glades::NNetworkEpochMetrics last;
	bool saw;
	int stopAfterEpoch;

	CaptureMetricsCb(int stopEpoch = 0)
	    : last(), saw(false), stopAfterEpoch(stopEpoch) {}

	virtual void onRunStart(const glades::NNetwork&, int) {}
	virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
	{
		last = m;
		saw = true;
		if (stopAfterEpoch > 0 && m.epoch >= stopAfterEpoch)
			return true;
		return false;
	}
	virtual void onRunEnd(const glades::NNetwork&, int) {}
};

// Build a simple NumberInput with synthetic "image" data for CNN classification.
// Each row is a flattened C*H*W image. Expected is one-hot encoded class label.
static glades::NumberInput* make_synthetic_image_data(
    unsigned int numSamples,
    unsigned int numClasses,
    unsigned int C, unsigned int H, unsigned int W,
    unsigned int seed)
{
	glades::NumberInput* di = new glades::NumberInput();
	const unsigned int featureCount = C * H * W;

	di->trainMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(featureCount, 0.0f));
	di->trainExpectedMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(numClasses, 0.0f));

	// Simple deterministic pseudo-random data generation.
	unsigned int s = seed;
	for (unsigned int i = 0; i < numSamples; ++i)
	{
		const unsigned int label = i % numClasses;
		for (unsigned int f = 0; f < featureCount; ++f)
		{
			// LCG for deterministic pseudo-random floats in [0, 1].
			s = s * 1103515245u + 12345u;
			float v = static_cast<float>((s >> 16) & 0x7FFFu) / 32767.0f;
			// Bias pixel values based on class label for learnability.
			if (f % numClasses == label)
				v = v * 0.5f + 0.5f; // brighter
			else
				v = v * 0.5f; // dimmer
			di->trainMatrix[i][f] = v;
		}
		di->trainExpectedMatrix[i][label] = 1.0f;
	}

	// Mirror train to test for simplicity.
	di->testMatrix = di->trainMatrix;
	di->testExpectedMatrix = di->trainExpectedMatrix;

	return di;
}

// Create a CNN NNetwork with a given config and return heap-allocated pointer.
// Caller owns the returned pointer and must delete it.
static glades::NNetwork* make_cnn(
    const char* name,
    unsigned int inputC, unsigned int inputH, unsigned int inputW,
    unsigned int numClasses,
    const std::vector<glades::CNNConfig::ConvLayerSpec>& convSpecs,
    const std::vector<unsigned int>& fcHiddenSizes,
    float lr,
    int batchSize,
    unsigned int seed)
{
	glades::InputLayerInfo* in = new glades::InputLayerInfo(
	    /*batchSize*/ batchSize,
	    /*learningRate*/ lr,
	    /*momentumFactor*/ 0.0f,
	    /*weightDecay1*/ 0.0f,
	    /*weightDecay2*/ 0.0f,
	    /*pDropout*/ 0.0f,
	    /*activationType*/ glades::GMath::RELU,
	    /*activationParam*/ 1.0f);

	std::vector<glades::HiddenLayerInfo*> hidden;
	for (size_t i = 0; i < fcHiddenSizes.size(); ++i)
	{
		hidden.push_back(new glades::HiddenLayerInfo(
		    static_cast<int>(fcHiddenSizes[i]),
		    lr,
		    0.0f, 0.0f, 0.0f, 0.0f,
		    glades::GMath::RELU,
		    1.0f));
	}

	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(
	    static_cast<int>(numClasses),
	    glades::OutputLayerInfo::CLASSIFICATION);

	glades::NNInfo* info = new glades::NNInfo(name, in, hidden, out);
	glades::NNetwork* net = new glades::NNetwork(info, glades::NNetwork::TYPE_CNN);
	net->setSeed(seed);

	// Configure CNN layers.
	glades::TrainingConfig& cfg = net->getTrainingConfigMutable();
	cfg.cnn.inputH = inputH;
	cfg.cnn.inputW = inputW;
	cfg.cnn.inputC = inputC;
	cfg.cnn.convLayers = convSpecs;

	delete info;
	return net;
}

// Helper: check that a float is finite (not NaN, not Inf).
static bool is_finite(float v)
{
	return (v == v) && (v - v == 0.0f);
}

} // anonymous namespace

void NNCNNUnitTest()
{
	printf("============================================================\n");
	printf("CNN Unit Test Suite\n");
	printf("============================================================\n");

	// ------------------------------------------------------------------
	// Test 1: Single conv layer, no BN, no pool - basic training smoke test
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 1: Single conv layer (no BN, no pool) - smoke test\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 42u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = false;
		spec.useMaxPool = false;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(16u);

		glades::NNetwork* net = make_cnn("ut_cnn_smoke", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.01f, 10, 123u);
		net->getTerminatorMutable().setEpoch(5);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN smoke train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::Smoke TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 2: Conv + MaxPool - training completes
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 2: Conv + MaxPool\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 77u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = false;
		spec.useMaxPool = true;
		spec.poolH = 2; spec.poolW = 2;
		spec.poolStrideH = 2; spec.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(8u);

		glades::NNetwork* net = make_cnn("ut_cnn_pool", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.01f, 10, 456u);
		net->getTerminatorMutable().setEpoch(5);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN pool train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::Pool TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 3: Conv + BatchNorm - training completes
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 3: Conv + BatchNorm\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 88u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = true;
		spec.useMaxPool = false;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(8u);

		glades::NNetwork* net = make_cnn("ut_cnn_bn", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.01f, 10, 789u);
		net->getTerminatorMutable().setEpoch(5);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN BN train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::BN TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 4: Conv + BN + MaxPool (full layer) - training completes
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 4: Conv + BN + MaxPool (full layer)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 99u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = true;
		spec.useMaxPool = true;
		spec.poolH = 2; spec.poolW = 2;
		spec.poolStrideH = 2; spec.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(8u);

		glades::NNetwork* net = make_cnn("ut_cnn_full_layer", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.01f, 10, 111u);
		net->getTerminatorMutable().setEpoch(5);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN full-layer train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::FullLayer TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 5: Two conv layers stacked - deeper architecture
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 5: Two conv layers stacked\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 3;
		const unsigned int numSamples = 30;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 222u);

		// Layer 1: conv 3x3, pad 1, pool 2x2
		glades::CNNConfig::ConvLayerSpec spec1;
		spec1.outChannels = 4;
		spec1.kernelH = 3; spec1.kernelW = 3;
		spec1.strideH = 1; spec1.strideW = 1;
		spec1.padH = 1; spec1.padW = 1;
		spec1.useBatchNorm = false;
		spec1.useMaxPool = true;
		spec1.poolH = 2; spec1.poolW = 2;
		spec1.poolStrideH = 2; spec1.poolStrideW = 2;

		// Layer 2: conv 3x3, pad 1, pool 2x2
		glades::CNNConfig::ConvLayerSpec spec2;
		spec2.outChannels = 8;
		spec2.kernelH = 3; spec2.kernelW = 3;
		spec2.strideH = 1; spec2.strideW = 1;
		spec2.padH = 1; spec2.padW = 1;
		spec2.useBatchNorm = false;
		spec2.useMaxPool = true;
		spec2.poolH = 2; spec2.poolW = 2;
		spec2.poolStrideH = 2; spec2.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec1);
		convSpecs.push_back(spec2);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(16u);

		glades::NNetwork* net = make_cnn("ut_cnn_2layer", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.005f, 10, 333u);
		net->getTerminatorMutable().setEpoch(5);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN 2-layer train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::2Layer TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 6: Multi-channel input (3 channels, like RGB)
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 6: Multi-channel input (3-channel RGB-like)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 3, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 444u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 8;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = true;
		spec.useMaxPool = true;
		spec.poolH = 2; spec.poolW = 2;
		spec.poolStrideH = 2; spec.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(16u);

		glades::NNetwork* net = make_cnn("ut_cnn_rgb", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.005f, 10, 555u);
		net->getTerminatorMutable().setEpoch(5);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN RGB train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::RGB TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 7: Loss decreases over training
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 7: Loss decreases over training\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 666u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = false;
		spec.useMaxPool = true;
		spec.poolH = 2; spec.poolW = 2;
		spec.poolStrideH = 2; spec.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(16u);

		// Train for 1 epoch, record loss.
		glades::NNetwork* net = make_cnn("ut_cnn_loss_dec", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.01f, 5, 777u);

		CaptureMetricsCb cb1(1);
		net->getTerminatorMutable().setEpoch(1000);
		net->getTerminatorMutable().setAccuracy(0);
		net->train(di, &cb1);
		G_assert(__FILE__, __LINE__, "==============CNN::LossDec Epoch1 Metrics Not Captured==============", cb1.saw);
		const float loss1 = cb1.last.totalError;
		printf("[UT] CNN loss after epoch 1: %f\n", loss1);
		G_assert(__FILE__, __LINE__, "==============CNN::LossDec Epoch1 Loss Not Finite==============", is_finite(loss1));

		// Train for 20 more epochs.
		CaptureMetricsCb cb2(21);
		net->train(di, &cb2);
		G_assert(__FILE__, __LINE__, "==============CNN::LossDec Epoch20 Metrics Not Captured==============", cb2.saw);
		const float loss2 = cb2.last.totalError;
		printf("[UT] CNN loss after epoch ~20: %f\n", loss2);
		G_assert(__FILE__, __LINE__, "==============CNN::LossDec Epoch20 Loss Not Finite==============", is_finite(loss2));

		// Loss should have decreased.
		printf("[UT] CNN loss comparison: epoch1=%f, epoch20=%f\n", loss1, loss2);
		G_assert(__FILE__, __LINE__, "==============CNN::LossDec Loss Did Not Decrease==============",
		         loss2 < loss1);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 8: Test (inference) mode returns valid status
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 8: Inference (test) mode\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 888u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = true;
		spec.useMaxPool = true;
		spec.poolH = 2; spec.poolW = 2;
		spec.poolStrideH = 2; spec.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(8u);

		glades::NNetwork* net = make_cnn("ut_cnn_infer", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.01f, 10, 999u);
		net->getTerminatorMutable().setEpoch(3);
		net->getTerminatorMutable().setAccuracy(0);

		// Train first.
		const glades::NNetworkStatus stTrain = net->train(di);
		G_assert(__FILE__, __LINE__, "==============CNN::Infer TrainStatus() Failed==============", stTrain.ok());

		// Now test (inference).
		const glades::NNetworkStatus stTest = net->test(di);
		printf("[UT] CNN test status: %s\n", stTest.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::Infer TestStatus() Failed==============", stTest.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 9: Save and load model - weight round-trip
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 9: Save and load model\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 1010u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = true;
		spec.useMaxPool = true;
		spec.poolH = 2; spec.poolW = 2;
		spec.poolStrideH = 2; spec.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(8u);

		glades::NNetwork* net1 = make_cnn("ut_cnn_save", C, H, W, numClasses,
		                                 convSpecs, fcHidden, 0.01f, 10, 1111u);
		net1->getTerminatorMutable().setEpoch(3);
		net1->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus stTrain = net1->train(di);
		G_assert(__FILE__, __LINE__, "==============CNN::SaveLoad Train() Failed==============", stTrain.ok());

		// Save the model.
		const glades::NNetworkStatus stSave = net1->saveModel("ut_cnn_roundtrip");
		printf("[UT] CNN save status: %s\n", stSave.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::SaveLoad SaveModel() Failed==============", stSave.ok());

		// Load into a fresh network.
		glades::NNetwork net2(glades::NNetwork::TYPE_CNN);
		const glades::NNetworkStatus stLoad = net2.loadModel("ut_cnn_roundtrip", di);
		printf("[UT] CNN load status: %s\n", stLoad.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::SaveLoad LoadModel() Failed==============", stLoad.ok());

		// Verify loaded network can run inference.
		const glades::NNetworkStatus stTest = net2.test(di);
		printf("[UT] CNN loaded-test status: %s\n", stTest.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::SaveLoad Test() After Load Failed==============", stTest.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net1;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 10: Multiple FC hidden layers after conv
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 10: Multiple FC hidden layers\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 1212u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = false;
		spec.useMaxPool = true;
		spec.poolH = 2; spec.poolW = 2;
		spec.poolStrideH = 2; spec.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		// Two FC hidden layers.
		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(16u);
		fcHidden.push_back(8u);

		glades::NNetwork* net = make_cnn("ut_cnn_multifc", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.01f, 10, 1313u);
		net->getTerminatorMutable().setEpoch(3);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN multi-FC train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::MultiFC TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 11: No FC hidden layers (conv -> directly to output)
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 11: No FC hidden layers (conv -> output)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 1414u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = false;
		spec.useMaxPool = true;
		spec.poolH = 2; spec.poolW = 2;
		spec.poolStrideH = 2; spec.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		// No FC hidden layers - flatten goes directly to output.
		std::vector<unsigned int> fcHidden;

		glades::NNetwork* net = make_cnn("ut_cnn_nofc", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.01f, 10, 1515u);
		net->getTerminatorMutable().setEpoch(3);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN no-FC train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::NoFC TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 12: 3-class classification
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 12: 3-class classification\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 3;
		const unsigned int numSamples = 30;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 1616u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 8;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = true;
		spec.useMaxPool = true;
		spec.poolH = 2; spec.poolW = 2;
		spec.poolStrideH = 2; spec.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(16u);

		glades::NNetwork* net = make_cnn("ut_cnn_3class", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.01f, 10, 1717u);
		net->getTerminatorMutable().setEpoch(5);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN 3-class train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::3Class TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 13: Stride > 1 (no padding)
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 13: Stride 2 convolution\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 1818u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 2; spec.strideW = 2;
		spec.padH = 0; spec.padW = 0;
		spec.useBatchNorm = false;
		spec.useMaxPool = false;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(8u);

		glades::NNetwork* net = make_cnn("ut_cnn_stride2", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.01f, 10, 1919u);
		net->getTerminatorMutable().setEpoch(3);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN stride2 train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::Stride2 TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 14: 5x5 kernel
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 14: 5x5 kernel\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 2020u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 5; spec.kernelW = 5;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 2; spec.padW = 2;
		spec.useBatchNorm = false;
		spec.useMaxPool = false;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(8u);

		glades::NNetwork* net = make_cnn("ut_cnn_5x5", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.01f, 10, 2121u);
		net->getTerminatorMutable().setEpoch(3);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN 5x5 train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::5x5 TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 15: AdamW optimizer
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 15: AdamW optimizer\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 2222u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = false;
		spec.useMaxPool = true;
		spec.poolH = 2; spec.poolW = 2;
		spec.poolStrideH = 2; spec.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(8u);

		glades::NNetwork* net = make_cnn("ut_cnn_adam", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.001f, 10, 2323u);

		// Enable AdamW.
		glades::TrainingConfig& tcfg = net->getTrainingConfigMutable();
		tcfg.optimizer.type = glades::OptimizerConfig::ADAMW;
		tcfg.optimizer.adamBeta1 = 0.9f;
		tcfg.optimizer.adamBeta2 = 0.999f;
		tcfg.optimizer.adamEps = 1e-8f;

		net->getTerminatorMutable().setEpoch(5);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN AdamW train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::AdamW TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 16: Deep CNN (3 conv layers + BN + pool on each)
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 16: Deep CNN (3 conv layers)\n");
	printf("-----------------------------------\n");
	{
		// Need bigger input for 3 pooling layers (each halves dims).
		// 16x16 -> 8x8 -> 4x4 -> 2x2
		const unsigned int C = 1, H = 16, W = 16;
		const unsigned int numClasses = 4;
		const unsigned int numSamples = 40;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 2424u);

		glades::CNNConfig::ConvLayerSpec spec1;
		spec1.outChannels = 4;
		spec1.kernelH = 3; spec1.kernelW = 3;
		spec1.strideH = 1; spec1.strideW = 1;
		spec1.padH = 1; spec1.padW = 1;
		spec1.useBatchNorm = true;
		spec1.useMaxPool = true;
		spec1.poolH = 2; spec1.poolW = 2;
		spec1.poolStrideH = 2; spec1.poolStrideW = 2;

		glades::CNNConfig::ConvLayerSpec spec2;
		spec2.outChannels = 8;
		spec2.kernelH = 3; spec2.kernelW = 3;
		spec2.strideH = 1; spec2.strideW = 1;
		spec2.padH = 1; spec2.padW = 1;
		spec2.useBatchNorm = true;
		spec2.useMaxPool = true;
		spec2.poolH = 2; spec2.poolW = 2;
		spec2.poolStrideH = 2; spec2.poolStrideW = 2;

		glades::CNNConfig::ConvLayerSpec spec3;
		spec3.outChannels = 16;
		spec3.kernelH = 3; spec3.kernelW = 3;
		spec3.strideH = 1; spec3.strideW = 1;
		spec3.padH = 1; spec3.padW = 1;
		spec3.useBatchNorm = true;
		spec3.useMaxPool = true;
		spec3.poolH = 2; spec3.poolW = 2;
		spec3.poolStrideH = 2; spec3.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec1);
		convSpecs.push_back(spec2);
		convSpecs.push_back(spec3);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(32u);

		glades::NNetwork* net = make_cnn("ut_cnn_deep", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.005f, 10, 2525u);
		net->getTerminatorMutable().setEpoch(3);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN deep train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::Deep TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 17: Save/load with BN running stats preserved
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 17: Save/load preserves BN running stats\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 2626u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = true;
		spec.useMaxPool = true;
		spec.poolH = 2; spec.poolW = 2;
		spec.poolStrideH = 2; spec.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(8u);

		glades::NNetwork* net1 = make_cnn("ut_cnn_bn_save", C, H, W, numClasses,
		                                 convSpecs, fcHidden, 0.01f, 10, 2727u);
		net1->getTerminatorMutable().setEpoch(5);
		net1->getTerminatorMutable().setAccuracy(0);

		G_assert(__FILE__, __LINE__, "==============CNN::BNSave Train() Failed==============", net1->train(di).ok());

		// Capture inference loss before save.
		CaptureMetricsCb cb1(1);
		net1->getTerminatorMutable().setEpoch(1000);
		net1->test(di, &cb1);
		G_assert(__FILE__, __LINE__, "==============CNN::BNSave Test() Failed==============", cb1.saw);
		const float testLoss1 = cb1.last.totalError;
		printf("[UT] CNN BN test loss (original): %f\n", testLoss1);

		// Save.
		G_assert(__FILE__, __LINE__, "==============CNN::BNSave SaveModel() Failed==============", net1->saveModel("ut_cnn_bn_roundtrip").ok());

		// Load into fresh net.
		glades::NNetwork net2(glades::NNetwork::TYPE_CNN);
		G_assert(__FILE__, __LINE__, "==============CNN::BNSave LoadModel() Failed==============", net2.loadModel("ut_cnn_bn_roundtrip", di).ok());

		// Inference on loaded network.
		CaptureMetricsCb cb2(1);
		net2.test(di, &cb2);
		G_assert(__FILE__, __LINE__, "==============CNN::BNSave Loaded Test() Failed==============", cb2.saw);
		const float testLoss2 = cb2.last.totalError;
		printf("[UT] CNN BN test loss (loaded): %f\n", testLoss2);

		// Losses should be identical (same weights, same BN running stats).
		const float tol = 1e-3f;
		printf("[UT] CNN BN loss diff: %f\n", fabs(testLoss1 - testLoss2));
		G_assert(__FILE__, __LINE__, "==============CNN::BNSave Loaded Loss Mismatch==============",
		         fabs(testLoss1 - testLoss2) < tol);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net1;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 18: Stochastic (batch=1) training works
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 18: Stochastic (batch=1) training\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 10;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 2828u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = false;
		spec.useMaxPool = true;
		spec.poolH = 2; spec.poolW = 2;
		spec.poolStrideH = 2; spec.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(8u);

		// Batch size = 1 (pure stochastic).
		glades::NNetwork* net = make_cnn("ut_cnn_sgd1", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.01f, 1, 2929u);
		net->getTerminatorMutable().setEpoch(3);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN stochastic train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::Stochastic TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 19: Non-square input (rectangular H != W)
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 19: Non-square input (6x10)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 6, W = 10;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 3030u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = false;
		spec.useMaxPool = true;
		spec.poolH = 2; spec.poolW = 2;
		spec.poolStrideH = 2; spec.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(8u);

		glades::NNetwork* net = make_cnn("ut_cnn_rect", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.01f, 10, 3131u);
		net->getTerminatorMutable().setEpoch(3);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN rect train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::Rect TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 20: Mixed conv layers (different kernel sizes per layer)
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 20: Mixed conv layers (3x3 then 1x1)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 3232u);

		// Layer 1: 3x3 conv with pool.
		glades::CNNConfig::ConvLayerSpec spec1;
		spec1.outChannels = 8;
		spec1.kernelH = 3; spec1.kernelW = 3;
		spec1.strideH = 1; spec1.strideW = 1;
		spec1.padH = 1; spec1.padW = 1;
		spec1.useBatchNorm = true;
		spec1.useMaxPool = true;
		spec1.poolH = 2; spec1.poolW = 2;
		spec1.poolStrideH = 2; spec1.poolStrideW = 2;

		// Layer 2: 1x1 conv (pointwise, no pool).
		glades::CNNConfig::ConvLayerSpec spec2;
		spec2.outChannels = 4;
		spec2.kernelH = 1; spec2.kernelW = 1;
		spec2.strideH = 1; spec2.strideW = 1;
		spec2.padH = 0; spec2.padW = 0;
		spec2.useBatchNorm = false;
		spec2.useMaxPool = false;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec1);
		convSpecs.push_back(spec2);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(8u);

		glades::NNetwork* net = make_cnn("ut_cnn_mixed", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.005f, 10, 3333u);
		net->getTerminatorMutable().setEpoch(3);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN mixed train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::Mixed TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 21: AdamW + BN + save/load + continued training
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 21: AdamW + BN continued training after load\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 3434u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = true;
		spec.useMaxPool = true;
		spec.poolH = 2; spec.poolW = 2;
		spec.poolStrideH = 2; spec.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(8u);

		glades::NNetwork* net1 = make_cnn("ut_cnn_adam_cont", C, H, W, numClasses,
		                                 convSpecs, fcHidden, 0.001f, 10, 3535u);

		glades::TrainingConfig& tcfg = net1->getTrainingConfigMutable();
		tcfg.optimizer.type = glades::OptimizerConfig::ADAMW;

		net1->getTerminatorMutable().setEpoch(3);
		net1->getTerminatorMutable().setAccuracy(0);

		G_assert(__FILE__, __LINE__, "==============CNN::AdamCont Train1() Failed==============", net1->train(di).ok());
		G_assert(__FILE__, __LINE__, "==============CNN::AdamCont SaveModel() Failed==============", net1->saveModel("ut_cnn_adam_cont").ok());

		// Load and continue training.
		glades::NNetwork net2(glades::NNetwork::TYPE_CNN);
		G_assert(__FILE__, __LINE__, "==============CNN::AdamCont LoadModel() Failed==============", net2.loadModel("ut_cnn_adam_cont", di).ok());

		// Set AdamW again on the loaded network.
		glades::TrainingConfig& tcfg2 = net2.getTrainingConfigMutable();
		tcfg2.optimizer.type = glades::OptimizerConfig::ADAMW;
		tcfg2.cnn = tcfg.cnn;

		net2.getTerminatorMutable().setEpoch(3);
		net2.getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st2 = net2.train(di);
		printf("[UT] CNN Adam continued train status: %s\n", st2.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::AdamCont Train2() Failed==============", st2.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net1;
		delete di;
	}

	// ------------------------------------------------------------------
	// Test 22: Large batch size (full batch = numSamples)
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN Test 22: Full batch training\n");
	printf("-----------------------------------\n");
	{
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 10;

		glades::NumberInput* di = make_synthetic_image_data(numSamples, numClasses, C, H, W, 3636u);

		glades::CNNConfig::ConvLayerSpec spec;
		spec.outChannels = 4;
		spec.kernelH = 3; spec.kernelW = 3;
		spec.strideH = 1; spec.strideW = 1;
		spec.padH = 1; spec.padW = 1;
		spec.useBatchNorm = false;
		spec.useMaxPool = true;
		spec.poolH = 2; spec.poolW = 2;
		spec.poolStrideH = 2; spec.poolStrideW = 2;

		std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
		convSpecs.push_back(spec);

		std::vector<unsigned int> fcHidden;
		fcHidden.push_back(8u);

		// Batch size = 0 means full batch.
		glades::NNetwork* net = make_cnn("ut_cnn_fullbatch", C, H, W, numClasses,
		                                convSpecs, fcHidden, 0.01f, 0, 3737u);
		net->getTerminatorMutable().setEpoch(3);
		net->getTerminatorMutable().setAccuracy(0);

		const glades::NNetworkStatus st = net->train(di);
		printf("[UT] CNN full-batch train status: %s\n", st.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNN::FullBatch TrainStatus() Failed==============", st.ok());

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
		delete net;
		delete di;
	}

	printf("============================================================\n");
	printf("CNN Unit Tests Completed\n");
	printf("============================================================\n");
}

void NNCNNMNISTUnitTest()
{
	printf("============================================================\n");
	printf("CNN MNIST Unit Test (full dataset: 60k train, 10k test)\n");
	printf("============================================================\n");

	// Load full MNIST dataset.
	glades::ImageInput* di = new glades::ImageInput();
	di->import(shmea::GString("MNIST"));
	// Cache all images in memory to avoid repeated PNG decode across epochs.
	di->rowCacheMaxEntries = di->getTrainSize() + di->getTestSize();
	printf("[UT] MNIST loaded: train=%u, test=%u, features=%u, cache=%u\n",
	       di->getTrainSize(), di->getTestSize(), di->getFeatureCount(),
	       di->rowCacheMaxEntries);
	G_assert(__FILE__, __LINE__, "==============CNNMNIST::TrainSize==============", di->getTrainSize() > 0);
	G_assert(__FILE__, __LINE__, "==============CNNMNIST::TestSize==============", di->getTestSize() > 0);

	// Build LeNet-like CNN:
	// Conv1: 8 filters 5x5, stride 1, pad 0 -> 24x24 -> MaxPool 2x2 -> 12x12
	// Conv2: 16 filters 5x5, stride 1, pad 0 -> 8x8 -> MaxPool 2x2 -> 4x4
	// Flatten: 16*4*4 = 256
	// FC hidden: 128
	// Output: 10

	const unsigned int inputC = 1, inputH = 28, inputW = 28;
	const unsigned int numClasses = 10;

	glades::CNNConfig::ConvLayerSpec conv1;
	conv1.outChannels = 8;
	conv1.kernelH = 5; conv1.kernelW = 5;
	conv1.strideH = 1; conv1.strideW = 1;
	conv1.padH = 0; conv1.padW = 0;
	conv1.useBatchNorm = false;
	conv1.useMaxPool = true;
	conv1.poolH = 2; conv1.poolW = 2;
	conv1.poolStrideH = 2; conv1.poolStrideW = 2;

	glades::CNNConfig::ConvLayerSpec conv2;
	conv2.outChannels = 16;
	conv2.kernelH = 5; conv2.kernelW = 5;
	conv2.strideH = 1; conv2.strideW = 1;
	conv2.padH = 0; conv2.padW = 0;
	conv2.useBatchNorm = false;
	conv2.useMaxPool = true;
	conv2.poolH = 2; conv2.poolW = 2;
	conv2.poolStrideH = 2; conv2.poolStrideW = 2;

	std::vector<glades::CNNConfig::ConvLayerSpec> convSpecs;
	convSpecs.push_back(conv1);
	convSpecs.push_back(conv2);

	std::vector<unsigned int> fcHidden;
	fcHidden.push_back(128u);

	glades::NNetwork* net = make_cnn("ut_cnn_mnist", inputC, inputH, inputW, numClasses,
	                                convSpecs, fcHidden, 0.001f, 64, 42u);

	// Configure AdamW + grad clipping.
	glades::TrainingConfig& tcfg = net->getTrainingConfigMutable();
	tcfg.optimizer.type = glades::OptimizerConfig::ADAMW;
	tcfg.optimizer.adamBeta1 = 0.9f;
	tcfg.optimizer.adamBeta2 = 0.999f;
	tcfg.optimizer.adamEps = 1e-8f;
	tcfg.globalGradClipNorm = 5.0f;

	// ------------------------------------------------------------------
	// Test A: Training completes & loss decreases
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN MNIST Test A: Training completes & loss decreases\n");
	printf("-----------------------------------\n");
	{
		// Train 1 epoch, capture loss.
		CaptureMetricsCb cb1(1);
		net->getTerminatorMutable().setEpoch(1000);
		net->getTerminatorMutable().setAccuracy(0);
		const glades::NNetworkStatus stA1 = net->train(di, &cb1);
		G_assert(__FILE__, __LINE__, "==============CNNMNIST::TestA Epoch1 Train Failed==============", stA1.ok());
		G_assert(__FILE__, __LINE__, "==============CNNMNIST::TestA Epoch1 Metrics==============", cb1.saw);
		const float loss1 = cb1.last.totalError;
		printf("[UT] MNIST loss after epoch 1: %f\n", loss1);
		G_assert(__FILE__, __LINE__, "==============CNNMNIST::TestA Epoch1 Finite==============", is_finite(loss1));

		// Train 4 more epochs.
		CaptureMetricsCb cb2(4);
		net->train(di, &cb2);
		G_assert(__FILE__, __LINE__, "==============CNNMNIST::TestA Epoch5 Metrics==============", cb2.saw);
		const float loss2 = cb2.last.totalError;
		printf("[UT] MNIST loss after epoch 5: %f\n", loss2);
		G_assert(__FILE__, __LINE__, "==============CNNMNIST::TestA Epoch5 Finite==============", is_finite(loss2));

		printf("[UT] MNIST loss comparison: epoch1=%f, epoch5=%f\n", loss1, loss2);
		G_assert(__FILE__, __LINE__, "==============CNNMNIST::TestA Loss Did Not Decrease==============",
		         loss2 < loss1);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
	}

	// ------------------------------------------------------------------
	// Test B: Test accuracy > 90%
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN MNIST Test B: Test accuracy > 80%%\n");
	printf("-----------------------------------\n");
	{
		CaptureMetricsCb cbTest(1);
		net->test(di, &cbTest);
		G_assert(__FILE__, __LINE__, "==============CNNMNIST::TestB Metrics==============", cbTest.saw);
		const float acc = cbTest.last.classAccuracy;
		printf("[UT] MNIST test accuracy: %.2f%%\n", acc);
		fflush(stdout);
		G_assert(__FILE__, __LINE__, "==============CNNMNIST::TestB Accuracy > 80%%==============",
		         acc > 80.0f);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
	}

	// ------------------------------------------------------------------
	// Test C: Save/load round-trip
	// ------------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("CNN MNIST Test C: Save/load round-trip\n");
	printf("-----------------------------------\n");
	{
		// Save trained model.
		const glades::NNetworkStatus stSave = net->saveModel("ut_cnn_mnist_roundtrip");
		printf("[UT] MNIST save status: %s\n", stSave.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNNMNIST::TestC SaveModel()==============", stSave.ok());

		// Load into fresh network.
		glades::NNetwork net2(glades::NNetwork::TYPE_CNN);
		const glades::NNetworkStatus stLoad = net2.loadModel("ut_cnn_mnist_roundtrip", di);
		printf("[UT] MNIST load status: %s\n", stLoad.message.c_str());
		G_assert(__FILE__, __LINE__, "==============CNNMNIST::TestC LoadModel()==============", stLoad.ok());

		// Inference on original.
		CaptureMetricsCb cbOrig(1);
		net->test(di, &cbOrig);
		G_assert(__FILE__, __LINE__, "==============CNNMNIST::TestC Orig Metrics==============", cbOrig.saw);

		// Inference on loaded.
		CaptureMetricsCb cbLoaded(1);
		net2.test(di, &cbLoaded);
		G_assert(__FILE__, __LINE__, "==============CNNMNIST::TestC Loaded Metrics==============", cbLoaded.saw);

		const float lossOrig = cbOrig.last.totalError;
		const float lossLoaded = cbLoaded.last.totalError;
		printf("[UT] MNIST round-trip loss: orig=%f, loaded=%f\n", lossOrig, lossLoaded);

		const float tol = 1e-3f;
		G_assert(__FILE__, __LINE__, "==============CNNMNIST::TestC Loss Mismatch==============",
		         fabs(lossOrig - lossLoaded) < tol);

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
	}

	printf("============================================================\n");
	printf("CNN MNIST Unit Tests Completed\n");
	printf("============================================================\n");

	delete net;
	delete di;
}
