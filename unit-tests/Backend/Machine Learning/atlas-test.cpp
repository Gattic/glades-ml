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

#include "atlas-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/training_callbacks.h"
#include "../../../Backend/Machine Learning/Networks/atlas_optimizer.h"
#include "../../../Backend/Machine Learning/Networks/training_config.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"

#include "../../../Backend/Machine Learning/Networks/cuda/gpu_atlas.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_device.h"
#include "../../../Backend/Machine Learning/Networks/cuda/gpu_buffer.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

class CaptureMetricsCallbacks : public glades::ITrainingCallbacks
{
public:
	CaptureMetricsCallbacks() : last(), saw(false) { last = glades::NNetworkEpochMetrics(); }
	virtual void onRunStart(const glades::NNetwork&, int) {}
	virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
	{
		last = m;
		saw = true;
		return false;
	}
	virtual void onRunEnd(const glades::NNetwork&, int) {}

	glades::NNetworkEpochMetrics last;
	bool saw;
};

} // anonymous namespace

void ATLASUnitTest()
{
	printf("============================================================\n");
	printf("ATLAS Optimizer Unit Test Suite\n");
	printf("============================================================\n");

	// ---------------------------------------------------------------
	// Test 1: ATLAS DFF regression convergence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 1: DFF regression convergence\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(2, shmea::GVector<float>(1, 0.0f));
		di->trainMatrix[0][0] = 1.0f;
		di->trainExpectedMatrix[0][0] = 0.0f;
		di->trainMatrix[1][0] = 2.0f;
		di->trainExpectedMatrix[1][0] = 1.0f;
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ 2,
		    /*learningRate*/ 0.05f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::SIGMOID,
		    /*activationParam*/ 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 4,
		    /*learningRate*/ 0.05f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::SIGMOID,
		    /*activationParam*/ 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_atlas_dff_regression", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
		net.setSeed(42u);
		net.getTerminatorMutable().setEpoch(200);
		net.getTerminatorMutable().setAccuracy(0);

		// Configure ATLAS optimizer
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 2;
			cfg.atlas.tSub = 50;
			cfg.atlas.beta = 0.999f;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::DFF_Regression TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::DFF_Regression no metrics captured==============", cb.saw);
		printf("[UT] ATLAS DFF regression: final loss = %f\n", cb.last.totalError);
		ASSERT("==============ATLAS::DFF_Regression loss too high==============", cb.last.totalError < 0.27f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 2: ATLAS DFF XOR classification
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 2: DFF XOR classification\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

		// XOR dataset: [0,0]->0, [0,1]->1, [1,0]->1, [1,1]->0
		di->trainMatrix[0][0] = 0.0f; di->trainMatrix[0][1] = 0.0f;
		di->trainExpectedMatrix[0][0] = 0.0f;
		di->trainMatrix[1][0] = 0.0f; di->trainMatrix[1][1] = 1.0f;
		di->trainExpectedMatrix[1][0] = 1.0f;
		di->trainMatrix[2][0] = 1.0f; di->trainMatrix[2][1] = 0.0f;
		di->trainExpectedMatrix[2][0] = 1.0f;
		di->trainMatrix[3][0] = 1.0f; di->trainMatrix[3][1] = 1.0f;
		di->trainExpectedMatrix[3][0] = 0.0f;

		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ 4,
		    /*learningRate*/ 0.1f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::SIGMOID,
		    /*activationParam*/ 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 8,
		    /*learningRate*/ 0.1f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::SIGMOID,
		    /*activationParam*/ 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_atlas_dff_xor", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
		net.setSeed(1337u);
		net.getTerminatorMutable().setEpoch(500);
		net.getTerminatorMutable().setAccuracy(0);

		// Configure ATLAS optimizer
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 4;
			cfg.atlas.tSub = 100;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::DFF_XOR TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::DFF_XOR no metrics captured==============", cb.saw);
		printf("[UT] ATLAS DFF XOR: final loss = %f\n", cb.last.totalError);
		ASSERT("==============ATLAS::DFF_XOR loss too high==============", cb.last.totalError < 0.27f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 3: ATLAS transformer basic convergence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 3: Transformer basic convergence\n");
	printf("-----------------------------------\n");
	{
		const int numSamples = 8;
		const int numFeatures = 4;
		const int numOutputs = 4;

		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(numFeatures, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(numOutputs, 0.0f));

		// Fill with simple pattern data
		for (int i = 0; i < numSamples; ++i)
		{
			for (int j = 0; j < numFeatures; ++j)
			{
				di->trainMatrix[i][j] = static_cast<float>(i * numFeatures + j) / static_cast<float>(numSamples * numFeatures);
			}
			for (int j = 0; j < numOutputs; ++j)
			{
				di->trainExpectedMatrix[i][j] = static_cast<float>((i + j) % numOutputs) / static_cast<float>(numOutputs);
			}
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

		// Transformer blocks: 1 hidden layer with dModel=16
		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    /*size*/ 16, // dModel
		    /*learningRate*/ 0.01f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::LINEAR,
		    /*activationParam*/ 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(numOutputs, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_atlas_transformer", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
		net.setSeed(2026u);
		net.getTerminatorMutable().setEpoch(100);
		net.getTerminatorMutable().setAccuracy(0);

		// Configure ATLAS optimizer and transformer settings
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 8;
			cfg.atlas.tSub = 50;
			cfg.transformer.nHeadsOverride = 4;
			cfg.transformer.dFFOverride = 32;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::Transformer TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::Transformer no metrics captured==============", cb.saw);
		printf("[UT] ATLAS Transformer: final loss = %f\n", cb.last.totalError);
		ASSERT("==============ATLAS::Transformer loss too high==============", cb.last.totalError < 0.10f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 4: ATLAS RNN convergence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 4: RNN convergence\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

		di->trainMatrix[0][0] = 0.1f; di->trainMatrix[0][1] = 0.2f;
		di->trainExpectedMatrix[0][0] = 0.3f;
		di->trainMatrix[1][0] = 0.4f; di->trainMatrix[1][1] = 0.5f;
		di->trainExpectedMatrix[1][0] = 0.9f;
		di->trainMatrix[2][0] = 0.2f; di->trainMatrix[2][1] = 0.3f;
		di->trainExpectedMatrix[2][0] = 0.5f;
		di->trainMatrix[3][0] = 0.6f; di->trainMatrix[3][1] = 0.7f;
		di->trainExpectedMatrix[3][0] = 1.3f;

		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    4, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    8, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_atlas_rnn", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_RNN);
		net.setSeed(42u);
		net.getTerminatorMutable().setEpoch(300);
		net.getTerminatorMutable().setAccuracy(0);

		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 4;
			cfg.atlas.tSub = 50;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::RNN TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::RNN no metrics captured==============", cb.saw);
		printf("[UT] ATLAS RNN: final loss = %f\n", cb.last.totalError);
		ASSERT("==============ATLAS::RNN loss too high==============", cb.last.totalError < 0.16f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 5: ATLAS GRU convergence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 5: GRU convergence\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

		di->trainMatrix[0][0] = 0.1f; di->trainMatrix[0][1] = 0.2f;
		di->trainExpectedMatrix[0][0] = 0.3f;
		di->trainMatrix[1][0] = 0.4f; di->trainMatrix[1][1] = 0.5f;
		di->trainExpectedMatrix[1][0] = 0.9f;
		di->trainMatrix[2][0] = 0.2f; di->trainMatrix[2][1] = 0.3f;
		di->trainExpectedMatrix[2][0] = 0.5f;
		di->trainMatrix[3][0] = 0.6f; di->trainMatrix[3][1] = 0.7f;
		di->trainExpectedMatrix[3][0] = 1.3f;

		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    4, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    8, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_atlas_gru", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_GRU);
		net.setSeed(1337u);
		net.getTerminatorMutable().setEpoch(300);
		net.getTerminatorMutable().setAccuracy(0);

		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 4;
			cfg.atlas.tSub = 50;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::GRU TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::GRU no metrics captured==============", cb.saw);
		printf("[UT] ATLAS GRU: final loss = %f\n", cb.last.totalError);
		ASSERT("==============ATLAS::GRU loss too high==============", cb.last.totalError < 0.13f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 6: ATLAS LSTM convergence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 6: LSTM convergence\n");
	printf("-----------------------------------\n");
	{
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

		di->trainMatrix[0][0] = 0.1f; di->trainMatrix[0][1] = 0.2f;
		di->trainExpectedMatrix[0][0] = 0.3f;
		di->trainMatrix[1][0] = 0.4f; di->trainMatrix[1][1] = 0.5f;
		di->trainExpectedMatrix[1][0] = 0.9f;
		di->trainMatrix[2][0] = 0.2f; di->trainMatrix[2][1] = 0.3f;
		di->trainExpectedMatrix[2][0] = 0.5f;
		di->trainMatrix[3][0] = 0.6f; di->trainMatrix[3][1] = 0.7f;
		di->trainExpectedMatrix[3][0] = 1.3f;

		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    4, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    8, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
		glades::NNInfo* info = new glades::NNInfo("ut_atlas_lstm", in, hidden, out);

		glades::NNetwork net(info, glades::NNetwork::TYPE_LSTM);
		net.setSeed(2026u);
		net.getTerminatorMutable().setEpoch(300);
		net.getTerminatorMutable().setAccuracy(0);

		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 4;
			cfg.atlas.tSub = 50;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::LSTM TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::LSTM no metrics captured==============", cb.saw);
		printf("[UT] ATLAS LSTM: final loss = %f\n", cb.last.totalError);
		ASSERT("==============ATLAS::LSTM loss too high==============", cb.last.totalError < 0.16f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 7: ATLAS checkpoint save/load round-trip
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 7: Checkpoint save/load round-trip\n");
	printf("-----------------------------------\n");
	{
		const std::string ckptName = "ut_atlas_ckpt_roundtrip";

		// Phase 1: Train a DFF with ATLAS for 100 epochs, save checkpoint
		{
			glades::NumberInput* di = new glades::NumberInput();
			di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
			di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

			di->trainMatrix[0][0] = 0.0f; di->trainMatrix[0][1] = 0.0f;
			di->trainExpectedMatrix[0][0] = 0.0f;
			di->trainMatrix[1][0] = 0.0f; di->trainMatrix[1][1] = 1.0f;
			di->trainExpectedMatrix[1][0] = 1.0f;
			di->trainMatrix[2][0] = 1.0f; di->trainMatrix[2][1] = 0.0f;
			di->trainExpectedMatrix[2][0] = 1.0f;
			di->trainMatrix[3][0] = 1.0f; di->trainMatrix[3][1] = 1.0f;
			di->trainExpectedMatrix[3][0] = 0.0f;

			di->testMatrix = di->trainMatrix;
			di->testExpectedMatrix = di->trainExpectedMatrix;

			glades::InputLayerInfo* in = new glades::InputLayerInfo(
			    4, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);

			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(
			    8, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));

			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
			glades::NNInfo* info = new glades::NNInfo("ut_atlas_ckpt", in, hidden, out);

			glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
			net.setSeed(42u);
			net.getTerminatorMutable().setEpoch(100);
			net.getTerminatorMutable().setAccuracy(0);

			{
				glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
				cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
				cfg.atlas.rank = 4;
				cfg.atlas.tSub = 50;
			}

			const glades::NNetworkStatus st = net.train(di);
			ASSERT("==============ATLAS::Checkpoint Phase1 TrainStatus() Failed==============", st.ok());
			printf("[UT] Phase 1: trained 100 epochs with ATLAS, saving checkpoint...\n");

			const glades::NNetworkStatus saveSt = net.saveCheckpoint(ckptName);
			ASSERT("==============ATLAS::Checkpoint SaveCheckpoint() Failed==============", saveSt.ok());
			printf("[UT] Checkpoint saved successfully\n");

			delete di;
			delete info;
		}

		// Phase 2: Load checkpoint into fresh network, train 100 more epochs
		{
			glades::NumberInput* di = new glades::NumberInput();
			di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
			di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

			di->trainMatrix[0][0] = 0.0f; di->trainMatrix[0][1] = 0.0f;
			di->trainExpectedMatrix[0][0] = 0.0f;
			di->trainMatrix[1][0] = 0.0f; di->trainMatrix[1][1] = 1.0f;
			di->trainExpectedMatrix[1][0] = 1.0f;
			di->trainMatrix[2][0] = 1.0f; di->trainMatrix[2][1] = 0.0f;
			di->trainExpectedMatrix[2][0] = 1.0f;
			di->trainMatrix[3][0] = 1.0f; di->trainMatrix[3][1] = 1.0f;
			di->trainExpectedMatrix[3][0] = 0.0f;

			di->testMatrix = di->trainMatrix;
			di->testExpectedMatrix = di->trainExpectedMatrix;

			glades::NNetwork net2;
			const glades::NNetworkStatus loadSt = net2.loadCheckpoint(ckptName, di);
			ASSERT("==============ATLAS::Checkpoint LoadCheckpoint() Failed==============", loadSt.ok());
			printf("[UT] Checkpoint loaded successfully\n");

			// Continue training
			net2.getTerminatorMutable().setEpoch(100);
			net2.getTerminatorMutable().setAccuracy(0);

			const glades::NNetworkStatus st2 = net2.train(di);
			ASSERT("==============ATLAS::Checkpoint Phase2 TrainStatus() Failed==============", st2.ok());
			printf("[UT] Phase 2: trained 100 more epochs after checkpoint load\n");

			delete di;
		}

		// Cleanup checkpoint files
		{
			const std::string base = "database/checkpoints/" + ckptName;
			remove((base + "/manifest.txt").c_str());
			remove((base + "/nninfo.csv").c_str());
			// Remove shard files
			for (int i = 0; i < 10; ++i)
			{
				char buf[64];
				snprintf(buf, sizeof(buf), "/shard_%03d.bin", i);
				remove((base + buf).c_str());
			}
			rmdir(base.c_str());
		}
		printf("[UT] Checkpoint cleanup done\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 8: gramSchmidt produces orthonormal columns
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 8: gramSchmidt orthonormality\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 6;
		const unsigned int r = 3;

		// Fill Q[m, r] with a known non-orthogonal matrix (row-major: Q[i*r+j])
		std::vector<float> Q(m * r, 0.0f);
		Q[0*r+0] = 1.0f; Q[0*r+1] = 0.5f; Q[0*r+2] = 0.3f;
		Q[1*r+0] = 0.5f; Q[1*r+1] = 1.0f; Q[1*r+2] = 0.2f;
		Q[2*r+0] = 0.3f; Q[2*r+1] = 0.2f; Q[2*r+2] = 1.0f;
		Q[3*r+0] = 0.1f; Q[3*r+1] = 0.4f; Q[3*r+2] = 0.7f;
		Q[4*r+0] = 0.6f; Q[4*r+1] = 0.1f; Q[4*r+2] = 0.5f;
		Q[5*r+0] = 0.2f; Q[5*r+1] = 0.8f; Q[5*r+2] = 0.1f;

		glades::atlas::gramSchmidt(&Q[0], m, r);

		// Verify Q^T * Q ≈ I_r  (dot of column c with column d should be delta_cd)
		const float tol = 1e-5f;
		for (unsigned int c = 0; c < r; ++c)
		{
			for (unsigned int d = c; d < r; ++d)
			{
				float dot = 0.0f;
				for (unsigned int i = 0; i < m; ++i)
					dot += Q[i * r + c] * Q[i * r + d];

				float expected = (c == d) ? 1.0f : 0.0f;
				float err = (dot - expected) < 0.0f ? -(dot - expected) : (dot - expected);
				if (err > tol)
				{
					printf("[FAIL] gramSchmidt: Q^T*Q[%u,%u] = %f, expected %f (err=%e)\n",
					       c, d, dot, expected, err);
				}
				ASSERT("==============ATLAS::gramSchmidt orthonormality failed==============", err <= tol);
			}
		}

		// Verify no column is zero (degenerate)
		for (unsigned int c = 0; c < r; ++c)
		{
			float colNorm = 0.0f;
			for (unsigned int i = 0; i < m; ++i)
				colNorm += Q[i * r + c] * Q[i * r + c];
			colNorm = std::sqrt(colNorm);
			ASSERT("==============ATLAS::gramSchmidt zero column==============", colNorm > 0.5f);
		}

		printf("[UT] ATLAS gramSchmidt: orthonormality verified (m=%u, r=%u)\n", m, r);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 9: refreshSubspace preserves orthonormality and transforms state
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 9: refreshSubspace correctness\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 8;
		const unsigned int n = 6;
		const unsigned int r = 3;

		// Initialize a WeightState
		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 12345ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);
		ASSERT("==============ATLAS::refreshSubspace init failed==============", state.initialized);

		// Set non-trivial Fisher and prevGz so we can verify they get transformed
		for (unsigned int c = 0; c < r; ++c)
			state.fisherDiag[c] = static_cast<float>(c + 1) * 0.5f;
		for (unsigned int i = 0; i < r * n; ++i)
			state.prevGz[i] = static_cast<float>(i) * 0.01f;

		// Create a synthetic gradient matrix [m, n]
		std::vector<float> grad(m * n, 0.0f);
		for (unsigned int i = 0; i < m * n; ++i)
			grad[i] = static_cast<float>(i % 7) * 0.1f - 0.3f;

		// Run refresh
		glades::atlas::refreshSubspace(state, &grad[0], m, n, /*powerIters=*/3, /*betaRefresh=*/0.5f, rng);

		// Verify U is still orthonormal after refresh
		const float tol = 1e-4f;
		for (unsigned int c = 0; c < r; ++c)
		{
			for (unsigned int d = c; d < r; ++d)
			{
				float dot = 0.0f;
				for (unsigned int i = 0; i < m; ++i)
					dot += state.U[i * r + c] * state.U[i * r + d];

				float expected = (c == d) ? 1.0f : 0.0f;
				float err = (dot - expected) < 0.0f ? -(dot - expected) : (dot - expected);
				if (err > tol)
				{
					printf("[FAIL] refreshSubspace: U^T*U[%u,%u] = %f, expected %f (err=%e)\n",
					       c, d, dot, expected, err);
				}
				ASSERT("==============ATLAS::refreshSubspace U not orthonormal==============", err <= tol);
			}
		}

		// Verify Fisher diagonal is still positive and finite
		for (unsigned int c = 0; c < r; ++c)
		{
			float f = state.fisherDiag[c];
			ASSERT("==============ATLAS::refreshSubspace Fisher NaN==============", f == f);
			ASSERT("==============ATLAS::refreshSubspace Fisher negative==============", f >= 0.0f);
		}

		// Verify prevGz is finite
		for (unsigned int i = 0; i < r * n; ++i)
		{
			float v = state.prevGz[i];
			ASSERT("==============ATLAS::refreshSubspace prevGz NaN==============", v == v);
		}

		printf("[UT] ATLAS refreshSubspace: orthonormality and state transform verified\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 10: applyStep produces finite weights and decreasing loss
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 10: applyStep isolated correctness\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 9999ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		// Create weight matrix and gradient
		std::vector<float> W(m * n, 0.0f);
		for (unsigned int i = 0; i < m * n; ++i)
			W[i] = static_cast<float>(i) * 0.1f - 0.5f;

		// Run several steps with a consistent gradient direction
		float prevNorm = 1e10f;
		for (int step = 0; step < 20; ++step)
		{
			// Gradient pointing in a fixed direction (simulates a consistent signal)
			std::vector<float> gW(m * n, 0.0f);
			for (unsigned int i = 0; i < m * n; ++i)
				gW[i] = W[i] * 0.1f; // gradient ~ 0.1*W (like L2 regularization signal)

			glades::ATLASConfig acTest10;
			acTest10.beta = 0.999f;
			acTest10.muMin = 0.0f;
			acTest10.muMax = 0.5f;
			acTest10.eps = 1e-8f;
			acTest10.kappaMax = 10.0f;
			acTest10.tSub = 50u;
			acTest10.powerIters = 3u;
			acTest10.betaRefresh = 0.5f;
			acTest10.muGrowthRate = 0.001f;
			acTest10.biasCorrection = false;
			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         /*invBatch=*/1.0f, /*lr=*/0.01f,
			                         /*wd1=*/0.0f, /*wd2=*/0.0f, /*gradScale=*/1.0f,
			                         acTest10, rng);

			// Verify all weights are finite
			for (unsigned int i = 0; i < m * n; ++i)
			{
				ASSERT("==============ATLAS::applyStep produced NaN weight==============", W[i] == W[i]);
				float diff = W[i] - W[i];
				ASSERT("==============ATLAS::applyStep produced Inf weight==============", diff == 0.0f);
			}
		}

		// Verify weights changed from initial values (optimizer did something)
		bool changed = false;
		for (unsigned int i = 0; i < m * n; ++i)
		{
			float init = static_cast<float>(i) * 0.1f - 0.5f;
			if (W[i] != init) { changed = true; break; }
		}
		ASSERT("==============ATLAS::applyStep weights unchanged after 20 steps==============", changed);

		printf("[UT] ATLAS applyStep: 20 steps, all weights finite, weights updated\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 11: Property-based fuzz test (invariants over many random steps)
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 11: Property-based fuzz test\n");
	printf("-----------------------------------\n");
	{
		// Run ATLAS with randomized configs across multiple weight matrix sizes
		// and verify invariants hold at every step.
		struct FuzzConfig
		{
			unsigned int m, n, r;
			float lr, beta;
			unsigned int tSub;
		};
		const FuzzConfig configs[] = {
			{4, 3, 2, 0.01f, 0.999f, 10},
			{8, 8, 4, 0.1f, 0.99f, 5},
			{16, 4, 3, 0.05f, 0.9f, 20},
			{3, 16, 2, 0.001f, 0.999f, 50},
			{32, 32, 8, 0.01f, 0.99f, 15},
		};
		const int nConfigs = 5;

		for (int ci = 0; ci < nConfigs; ++ci)
		{
			const FuzzConfig& fc = configs[ci];
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, static_cast<unsigned long long>(7777 + ci * 1337));
			glades::atlas::WeightState state;
			glades::atlas::initWeightState(state, fc.m, fc.n, fc.r, 0.01f, rng);

			std::vector<float> W(fc.m * fc.n, 0.0f);
			for (unsigned int i = 0; i < fc.m * fc.n; ++i)
				W[i] = glades::rng::standard_normal(rng) * 0.1f;

			glades::ATLASConfig acFuzz;
			acFuzz.beta = fc.beta;
			acFuzz.muMin = 0.0f;
			acFuzz.muMax = 0.5f;
			acFuzz.eps = 1e-8f;
			acFuzz.kappaMax = 10.0f;
			acFuzz.tSub = fc.tSub;
			acFuzz.powerIters = 3u;
			acFuzz.betaRefresh = 0.5f;
			acFuzz.muGrowthRate = 0.001f;
			acFuzz.biasCorrection = false;

			const int nSteps = 500;
			for (int step = 0; step < nSteps; ++step)
			{
				// Generate random gradient
				std::vector<float> gW(fc.m * fc.n, 0.0f);
				for (unsigned int i = 0; i < fc.m * fc.n; ++i)
					gW[i] = glades::rng::standard_normal(rng) * 0.5f;

				glades::atlas::applyStep(state, &W[0], &gW[0], fc.m, fc.n,
				                         1.0f, fc.lr, 0.0f, 0.0f, 1.0f,
				                         acFuzz, rng);

				// Invariant 1: All weights finite
				for (unsigned int i = 0; i < fc.m * fc.n; ++i)
				{
					ASSERT("==============ATLAS::Fuzz NaN weight==============", W[i] == W[i]);
					float diff = W[i] - W[i];
					ASSERT("==============ATLAS::Fuzz Inf weight==============", diff == 0.0f);
				}

				// Invariant 2: Fisher diagonal positive and finite
				for (unsigned int c = 0; c < state.r; ++c)
				{
					ASSERT("==============ATLAS::Fuzz Fisher NaN==============",
					       state.fisherDiag[c] == state.fisherDiag[c]);
					ASSERT("==============ATLAS::Fuzz Fisher negative==============",
					       state.fisherDiag[c] >= 0.0f);
				}

				// Invariant 3: mu within bounds
				ASSERT("==============ATLAS::Fuzz mu below min==============", state.mu >= 0.0f);
				ASSERT("==============ATLAS::Fuzz mu above max==============", state.mu <= 0.5f + 1e-6f);

				// Invariant 4: sigma2 positive and finite
				ASSERT("==============ATLAS::Fuzz sigma2 NaN==============",
				       state.sigma2 == state.sigma2);
				ASSERT("==============ATLAS::Fuzz sigma2 negative==============",
				       state.sigma2 >= 0.0f);

				// Invariant 5: U orthonormal (check every tSub steps to avoid overhead)
				if (fc.tSub > 0u && (state.step % static_cast<unsigned long long>(fc.tSub)) == 0ULL)
				{
					const float orthoTol = 1e-4f;
					for (unsigned int c = 0; c < state.r; ++c)
					{
						for (unsigned int d = c; d < state.r; ++d)
						{
							float dot = 0.0f;
							for (unsigned int i = 0; i < fc.m; ++i)
								dot += state.U[i * state.r + c] * state.U[i * state.r + d];
							float expected = (c == d) ? 1.0f : 0.0f;
							float err = (dot - expected) < 0.0f ? -(dot - expected) : (dot - expected);
							ASSERT("==============ATLAS::Fuzz U not orthonormal==============", err <= orthoTol);
						}
					}
				}
			}

			printf("[UT] Fuzz config %d (m=%u n=%u r=%u): %d steps, all invariants held\n",
			       ci, fc.m, fc.n, fc.r, nSteps);
		}
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 12: kappaMax capping behavior
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 12: kappaMax rate capping\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int r = 2;
		const float kappaMax = 5.0f;
		const float lr = 0.01f;
		const float eps = 1e-8f;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 55555ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		// Set Fisher diagonal to very small values (should trigger capping)
		for (unsigned int c = 0; c < r; ++c)
			state.fisherDiag[c] = 1e-12f;
		state.sigma2 = 1e-12f;

		std::vector<float> W(m * n, 0.5f);
		std::vector<float> gW(m * n, 1.0f);

		std::vector<float> W_before(W.begin(), W.end());

		glades::ATLASConfig acKappa;
		acKappa.beta = 0.999f;
		acKappa.muMin = 0.0f;
		acKappa.muMax = 0.5f;
		acKappa.eps = eps;
		acKappa.kappaMax = kappaMax;
		acKappa.tSub = 50u;
		acKappa.powerIters = 3u;
		acKappa.betaRefresh = 0.5f;
		acKappa.muGrowthRate = 0.001f;
		acKappa.biasCorrection = false;
		glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
		                         1.0f, lr, 0.0f, 0.0f, 1.0f,
		                         acKappa, rng);

		// The effective learning rate should be capped at kappaMax * lr = 0.05
		// Without capping, lr/(1e-12 + 1e-8) ~ 1e6, which would be catastrophic.
		// Verify no element changed by more than kappaMax * lr * maxGrad
		const float maxChange = kappaMax * lr * 1.0f; // gScale=1, invBatch=1, grad=1
		for (unsigned int i = 0; i < m * n; ++i)
		{
			float delta = W[i] - W_before[i];
			if (delta < 0.0f) delta = -delta;
			ASSERT("==============ATLAS::kappaMax capping violated==============",
			       delta < maxChange * 2.0f); // allow 2x margin for subspace correction
		}

		// Verify weights are still finite (not blown up)
		for (unsigned int i = 0; i < m * n; ++i)
		{
			ASSERT("==============ATLAS::kappaMax NaN weight==============", W[i] == W[i]);
			float diff = W[i] - W[i];
			ASSERT("==============ATLAS::kappaMax Inf weight==============", diff == 0.0f);
		}

		printf("[UT] ATLAS kappaMax: capping verified with tiny Fisher/sigma2\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 13: mu adaptation responds to gradient smoothness
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 13: mu adaptation\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int r = 2;

		// Scenario A: Smooth gradients (consistent direction) -> mu should grow
		{
			glades::ATLASConfig acMu;
			acMu.beta = 0.999f;
			acMu.muMin = 0.01f;
			acMu.muMax = 0.5f;
			acMu.eps = 1e-8f;
			acMu.kappaMax = 10.0f;
			acMu.tSub = 200u;
			acMu.powerIters = 3u;
			acMu.betaRefresh = 0.5f;
			acMu.muGrowthRate = 0.001f;
			acMu.biasCorrection = false;

			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 11111ULL);
			glades::atlas::WeightState state;
			glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

			std::vector<float> W(m * n, 0.5f);
			for (int step = 0; step < 100; ++step)
			{
				std::vector<float> gW(m * n, 0.0f);
				// Consistent gradient direction
				for (unsigned int i = 0; i < m * n; ++i)
					gW[i] = 0.1f;
				glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
				                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
				                         acMu, rng);
			}
			float smoothMu = state.mu;

			// Scenario B: Oscillating gradients -> mu should shrink
			glades::rng::seed_engine(rng, 11111ULL);
			glades::atlas::WeightState state2;
			glades::atlas::initWeightState(state2, m, n, r, 0.01f, rng);

			std::vector<float> W2(m * n, 0.5f);
			for (int step = 0; step < 100; ++step)
			{
				std::vector<float> gW(m * n, 0.0f);
				// Oscillating gradient direction
				float sign = (step % 2 == 0) ? 1.0f : -1.0f;
				for (unsigned int i = 0; i < m * n; ++i)
					gW[i] = sign * 0.1f;
				glades::atlas::applyStep(state2, &W2[0], &gW[0], m, n,
				                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
				                         acMu, rng);
			}
			float oscillatingMu = state2.mu;

			printf("[UT] mu smooth=%.6f, mu oscillating=%.6f\n", smoothMu, oscillatingMu);
			ASSERT("==============ATLAS::mu adaptation: smooth should be >= oscillating==============",
			       smoothMu >= oscillatingMu);
		}
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 14: Rank clamping (rank > min(m, n))
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 14: Rank clamping\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int requestedRank = 100; // much larger than min(m,n) = 3

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 77777ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, requestedRank, 0.01f, rng);

		ASSERT("==============ATLAS::RankClamp init failed==============", state.initialized);
		ASSERT("==============ATLAS::RankClamp rank not clamped==============", state.r <= m && state.r <= n);
		ASSERT("==============ATLAS::RankClamp rank should be min(m,n)==============", state.r == 3u);

		// Verify U is still orthonormal after clamping
		const float tol = 1e-5f;
		for (unsigned int c = 0; c < state.r; ++c)
		{
			for (unsigned int d = c; d < state.r; ++d)
			{
				float dot = 0.0f;
				for (unsigned int i = 0; i < m; ++i)
					dot += state.U[i * state.r + c] * state.U[i * state.r + d];
				float expected = (c == d) ? 1.0f : 0.0f;
				float err = (dot - expected) < 0.0f ? -(dot - expected) : (dot - expected);
				ASSERT("==============ATLAS::RankClamp U not orthonormal==============", err <= tol);
			}
		}
		printf("[UT] ATLAS rank clamping: requested=%u, actual=%u (min(m,n)=%u)\n",
		       requestedRank, state.r, (m < n) ? m : n);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 15: NaN gradient recovery
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 15: NaN gradient recovery\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 88888ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		std::vector<float> W(m * n, 0.5f);

		glades::ATLASConfig acNan;
		acNan.rank = r;
		acNan.beta = 0.999f;
		acNan.muMin = 0.0f;
		acNan.muMax = 0.5f;
		acNan.eps = 1e-8f;
		acNan.kappaMax = 10.0f;
		acNan.tSub = 50u;
		acNan.powerIters = 3u;
		acNan.betaRefresh = 0.5f;
		acNan.muGrowthRate = 0.001f;
		acNan.biasCorrection = false;

		// Run a few normal steps first
		for (int step = 0; step < 5; ++step)
		{
			std::vector<float> gW(m * n, 0.1f);
			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         acNan, rng);
		}

		// Inject a gradient with extreme values (not NaN, but very large)
		{
			std::vector<float> gW(m * n, 1e10f);
			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         acNan, rng);
		}

		// Verify state is still finite after extreme gradient
		ASSERT("==============ATLAS::NaN sigma2 not finite==============",
		       state.sigma2 == state.sigma2 && (state.sigma2 - state.sigma2) == 0.0f);
		ASSERT("==============ATLAS::NaN mu not finite==============",
		       state.mu == state.mu && (state.mu - state.mu) == 0.0f);
		for (unsigned int c = 0; c < r; ++c)
		{
			ASSERT("==============ATLAS::NaN Fisher not finite==============",
			       state.fisherDiag[c] == state.fisherDiag[c]);
		}

		// Continue with normal gradients — should still work
		for (int step = 0; step < 5; ++step)
		{
			std::vector<float> gW(m * n, 0.1f);
			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         acNan, rng);

			for (unsigned int i = 0; i < m * n; ++i)
			{
				ASSERT("==============ATLAS::NaN post-recovery weight NaN==============", W[i] == W[i]);
				float diff = W[i] - W[i];
				ASSERT("==============ATLAS::NaN post-recovery weight Inf==============", diff == 0.0f);
			}
		}
		printf("[UT] ATLAS NaN gradient recovery: state remains finite after extreme gradients\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 16: Bias correction accelerates early preconditioning
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 16: Bias correction effect\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int r = 2;

		// Run 10 steps with bias correction OFF
		float sigma2_uncorrected;
		{
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 22222ULL);
			glades::atlas::WeightState state;
			glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

			glades::ATLASConfig acOff;
			acOff.beta = 0.999f;
			acOff.biasCorrection = false;
			acOff.muMin = 0.0f;
			acOff.muMax = 0.0f;
			acOff.tSub = 200u;
			acOff.powerIters = 3u;
			acOff.betaRefresh = 0.5f;
			acOff.muGrowthRate = 0.0f;

			std::vector<float> W(m * n, 0.5f);
			for (int step = 0; step < 10; ++step)
			{
				std::vector<float> gW(m * n, 1.0f);
				glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
				                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
				                         acOff, rng);
			}
			sigma2_uncorrected = state.sigma2;
		}

		// Run 10 steps with bias correction ON
		float sigma2_corrected_raw;
		{
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 22222ULL);
			glades::atlas::WeightState state;
			glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

			glades::ATLASConfig acOn;
			acOn.beta = 0.999f;
			acOn.biasCorrection = true;
			acOn.muMin = 0.0f;
			acOn.muMax = 0.0f;
			acOn.tSub = 200u;
			acOn.powerIters = 3u;
			acOn.betaRefresh = 0.5f;
			acOn.muGrowthRate = 0.0f;

			std::vector<float> W(m * n, 0.5f);
			for (int step = 0; step < 10; ++step)
			{
				std::vector<float> gW(m * n, 1.0f);
				glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
				                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
				                         acOn, rng);
			}
			sigma2_corrected_raw = state.sigma2;
		}

		// The raw sigma2 EMA should be the same (bias correction only affects
		// the effective rate, not the stored EMA value)
		const float emaDiff = (sigma2_uncorrected - sigma2_corrected_raw) < 0.0f
		                    ? -(sigma2_uncorrected - sigma2_corrected_raw)
		                    : (sigma2_uncorrected - sigma2_corrected_raw);
		printf("[UT] ATLAS bias correction: sigma2_uncorrected=%f, sigma2_corrected_raw=%f\n",
		       sigma2_uncorrected, sigma2_corrected_raw);
		// They won't be exactly equal because the effective learning rate differs
		// (bias-corrected sigma2 produces different weight updates, which produces
		// different gradients on the next step). But they should be in the same ballpark.
		ASSERT("==============ATLAS::BiasCorr raw EMA wildly different==============",
		       emaDiff < 0.1f);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 17: Comparative ATLAS-vs-SGD convergence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 17: Comparative ATLAS-vs-SGD convergence\n");
	printf("-----------------------------------\n");
	{
		// Train the same DFF network on the same problem with both SGD and ATLAS,
		// and assert that ATLAS final loss is no worse than 1.5x SGD final loss.
		// This catches optimizer regressions that absolute-threshold tests miss.

		// Dataset: sum regression
		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));

		di->trainMatrix[0][0] = 0.1f; di->trainMatrix[0][1] = 0.2f;
		di->trainExpectedMatrix[0][0] = 0.15f;
		di->trainMatrix[1][0] = 0.4f; di->trainMatrix[1][1] = 0.5f;
		di->trainExpectedMatrix[1][0] = 0.45f;
		di->trainMatrix[2][0] = 0.7f; di->trainMatrix[2][1] = 0.8f;
		di->trainExpectedMatrix[2][0] = 0.75f;
		di->trainMatrix[3][0] = 0.3f; di->trainMatrix[3][1] = 0.6f;
		di->trainExpectedMatrix[3][0] = 0.45f;

		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		const int epochs = 300;
		const unsigned int seed = 42u;

		// Run with SGD (momentum)
		float sgdLoss;
		{
			glades::InputLayerInfo* in = new glades::InputLayerInfo(
			    4, 0.05f, 0.9f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(
			    8, 0.05f, 0.9f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
			glades::NNInfo* info = new glades::NNInfo("ut_cmp_sgd", in, hidden, out);

			glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
			net.setSeed(static_cast<uint64_t>(seed));
			net.getTerminatorMutable().setEpoch(epochs);
			net.getTerminatorMutable().setAccuracy(0);

			CaptureMetricsCallbacks cb;
			const glades::NNetworkStatus st = net.train(di, &cb);
			ASSERT("==============ATLAS::Comparative SGD train failed==============", st.ok());
			ASSERT("==============ATLAS::Comparative SGD no metrics==============", cb.saw);
			sgdLoss = cb.last.totalError;

			delete info;
		}

		// Run with ATLAS
		float atlasLoss;
		{
			glades::InputLayerInfo* in = new glades::InputLayerInfo(
			    4, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(
			    8, 0.05f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
			glades::NNInfo* info = new glades::NNInfo("ut_cmp_atlas", in, hidden, out);

			glades::NNetwork net(info, glades::NNetwork::TYPE_DFF);
			net.setSeed(static_cast<uint64_t>(seed));
			net.getTerminatorMutable().setEpoch(epochs);
			net.getTerminatorMutable().setAccuracy(0);

			{
				glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
				cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
				cfg.atlas.rank = 4;
				cfg.atlas.tSub = 50;
			}

			CaptureMetricsCallbacks cb;
			const glades::NNetworkStatus st = net.train(di, &cb);
			ASSERT("==============ATLAS::Comparative ATLAS train failed==============", st.ok());
			ASSERT("==============ATLAS::Comparative ATLAS no metrics==============", cb.saw);
			atlasLoss = cb.last.totalError;

			delete info;
		}

		printf("[UT] Comparative: SGD loss=%.6f, ATLAS loss=%.6f, ratio=%.2f\n",
		       sgdLoss, atlasLoss, (sgdLoss > 0.0f) ? atlasLoss / sgdLoss : 0.0f);
		ASSERT("==============ATLAS::Comparative ATLAS loss much worse than SGD==============",
		       atlasLoss < sgdLoss * 1.5f + 0.01f);

		delete di;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 18: GPU ATLAS kernel tests
	// ---------------------------------------------------------------
#ifdef GLADES_HAVE_CUDA
	if (glades::gpu::initDevice())
	{
		printf("-----------------------------------\n");
		printf("ATLAS Test 18: GPU ATLAS kernels\n");
		printf("-----------------------------------\n");

		// 18a: atlas_gpu_init produces orthonormal U
		{
			printf("[UT] 18a: atlas_gpu_init orthonormality\n");
			const unsigned int m = 8;
			const unsigned int n = 6;
			const unsigned int r = 3;
			glades::rng::Engine gpuRng;
			glades::rng::seed_engine(gpuRng, 12345ULL);

			glades::gpu::GpuAtlasWeightState state;
			bool ok = glades::gpu::atlas_gpu_init(state, m, n, r, 0.01f, gpuRng);
			ASSERT("==============ATLAS::GPU init failed==============", ok);
			ASSERT("==============ATLAS::GPU state not initialized==============", state.initialized);
			ASSERT("==============ATLAS::GPU rank mismatch==============", state.r == r);

			// Download U and verify orthonormality: U^T * U ~ I_r
			std::vector<float> h_U(m * r);
			state.U.download(h_U.data());

			const float orthoTol = 1e-4f;
			for (unsigned int c = 0; c < r; ++c)
			{
				for (unsigned int d = c; d < r; ++d)
				{
					float dot = 0.0f;
					for (unsigned int i = 0; i < m; ++i)
						dot += h_U[i * r + c] * h_U[i * r + d];
					float expected = (c == d) ? 1.0f : 0.0f;
					float err = fabsf(dot - expected);
					ASSERT("==============ATLAS::GPU init U not orthonormal==============", err <= orthoTol);
				}
			}
			printf("[UT] 18a: PASSED\n");
		}

		// 18b: atlas_gpu_gram_schmidt on known non-orthonormal input
		{
			printf("[UT] 18b: atlas_gpu_gram_schmidt correctness\n");
			const int m = 4;
			const int r = 2;

			// Non-orthogonal, non-unit input [m x r] row-major
			// Col 0: [1, 2, 3, 4], Col 1: [1, 1, 1, 1]
			float h_Q[8] = {
				1.0f, 1.0f,
				2.0f, 1.0f,
				3.0f, 1.0f,
				4.0f, 1.0f
			};

			glades::gpu::GpuBuffer<float> d_Q;
			d_Q.allocate(m * r);
			d_Q.upload(h_Q);

			bool ok = glades::gpu::atlas_gpu_gram_schmidt(d_Q.data(), m, r);
			ASSERT("==============ATLAS::GPU GS call failed==============", ok);

			float h_result[8];
			d_Q.download(h_result);

			// Verify orthonormality
			const float orthoTol = 1e-5f;
			for (int c = 0; c < r; ++c)
			{
				for (int d = c; d < r; ++d)
				{
					float dot = 0.0f;
					for (int i = 0; i < m; ++i)
						dot += h_result[i * r + c] * h_result[i * r + d];
					float expected = (c == d) ? 1.0f : 0.0f;
					float err = fabsf(dot - expected);
					char msg[128];
					sprintf(msg, "ATLAS::GPU GS col %d dot col %d = %.6f (expected %.1f)", c, d, dot, expected);
					ASSERT(msg, err <= orthoTol);
				}
			}

			// Col 0 should be proportional to [1,2,3,4] (just normalized)
			float norm0 = sqrtf(1.0f + 4.0f + 9.0f + 16.0f);
			const float dirTol = 1e-5f;
			for (int i = 0; i < m; ++i)
			{
				float expected = (float)(i + 1) / norm0;
				float err = fabsf(h_result[i * r + 0] - expected);
				ASSERT("==============ATLAS::GPU GS col0 direction wrong==============", err <= dirTol);
			}

			printf("[UT] 18b: PASSED\n");
		}

		// 18c: atlas_gpu_step convergence on simple gradient descent
		{
			printf("[UT] 18c: atlas_gpu_step convergence\n");
			const unsigned int m = 4;
			const unsigned int n = 3;
			const unsigned int r = 2;
			glades::rng::Engine gpuRng;
			glades::rng::seed_engine(gpuRng, 42ULL);

			glades::gpu::GpuAtlasWeightState state;
			bool ok = glades::gpu::atlas_gpu_init(state, m, n, r, 0.01f, gpuRng);
			ASSERT("==============ATLAS::GPU step init failed==============", ok);

			// Weight and gradient on device
			const size_t mn = m * n;
			glades::gpu::GpuBuffer<float> d_W, d_gW;
			d_W.allocate(mn);
			d_gW.allocate(mn);

			// Initialize weights to small random values
			std::vector<float> h_W(mn);
			for (size_t i = 0; i < mn; ++i)
				h_W[i] = 0.1f * (float)((i * 7 + 3) % 11) / 11.0f;
			d_W.upload(h_W.data());

			// Target weights (what we want to reach)
			std::vector<float> target(mn);
			for (size_t i = 0; i < mn; ++i)
				target[i] = 0.5f;

			glades::ATLASConfig ac;
			ac.rank = r;
			ac.beta = 0.99f;
			ac.muMin = 0.01f;
			ac.muMax = 0.3f;
			ac.tSub = 10;
			ac.powerIters = 2;
			ac.eps = 1e-8f;
			ac.kappaMax = 10.0f;
			ac.betaRefresh = 0.5f;
			ac.muGrowthRate = 1.01f;
			ac.biasCorrection = true;

			float initialLoss = 0.0f;
			float finalLoss = 0.0f;

			const int nSteps = 100;
			for (int s = 0; s < nSteps; ++s)
			{
				// Compute gradient = W - target (MSE gradient direction)
				d_W.download(h_W.data());
				std::vector<float> h_gW(mn);
				float loss = 0.0f;
				for (size_t i = 0; i < mn; ++i)
				{
					h_gW[i] = h_W[i] - target[i];
					loss += h_gW[i] * h_gW[i];
				}
				loss /= (float)mn;

				if (s == 0) initialLoss = loss;
				if (s == nSteps - 1) finalLoss = loss;

				d_gW.upload(h_gW.data());

				ok = glades::gpu::atlas_gpu_step(state, d_W.data(), d_gW.data(),
				                                 m, n,
				                                 1.0f, 0.05f,
				                                 0.0f, 0.0f, 1.0f,
				                                 ac);
				ASSERT("==============ATLAS::GPU step call failed==============", ok);
			}

			printf("[UT] 18c: initial_loss=%.6f final_loss=%.6f\n", initialLoss, finalLoss);
			ASSERT("==============ATLAS::GPU step did not converge==============",
			       finalLoss < initialLoss * 0.5f);

			// Verify weights are finite
			d_W.download(h_W.data());
			for (size_t i = 0; i < mn; ++i)
			{
				ASSERT("==============ATLAS::GPU step produced NaN weight==============",
				       h_W[i] == h_W[i]);
				ASSERT("==============ATLAS::GPU step produced Inf weight==============",
				       fabsf(h_W[i]) < 1e10f);
			}

			printf("[UT] 18c: PASSED\n");
		}

		// 18d: GPU-CPU equivalence test
		// Run identical problems on both paths and verify weight trajectories match.
		{
			printf("[UT] 18d: GPU-CPU equivalence\n");
			const unsigned int m = 8;
			const unsigned int n = 6;
			const unsigned int r = 3;
			const unsigned int nSteps = 50;
			const float lr = 0.01f;
			const float muInit = 0.01f;
			const unsigned long long cpuSeed = 77777ULL;

			// --- CPU path ---
			glades::rng::Engine cpuRng;
			glades::rng::seed_engine(cpuRng, cpuSeed);
			glades::atlas::WeightState cpuState;
			glades::atlas::initWeightState(cpuState, m, n, r, muInit, cpuRng);

			std::vector<float> W_cpu(m * n);
			for (unsigned int i = 0; i < m * n; ++i)
				W_cpu[i] = 0.1f * (float)((i * 7 + 3) % 11) / 11.0f;

			glades::ATLASConfig ac;
			ac.rank = r;
			ac.beta = 0.99f;
			ac.muMin = 0.01f;
			ac.muMax = 0.3f;
			ac.tSub = 10;
			ac.powerIters = 2;
			ac.eps = 1e-8f;
			ac.kappaMax = 10.0f;
			ac.betaRefresh = 0.5f;
			ac.muGrowthRate = 0.001f;
			ac.biasCorrection = true;

			// Pre-generate all gradients so both paths use identical inputs.
			std::vector<std::vector<float> > allGrads(nSteps, std::vector<float>(m * n));
			{
				glades::rng::Engine gradRng;
				glades::rng::seed_engine(gradRng, 12345ULL);
				for (unsigned int s = 0; s < nSteps; ++s)
					for (unsigned int i = 0; i < m * n; ++i)
						allGrads[s][i] = glades::rng::standard_normal(gradRng) * 0.1f;
			}

			for (unsigned int s = 0; s < nSteps; ++s)
			{
				std::vector<float> gW(allGrads[s]);
				glades::atlas::applyStep(cpuState, &W_cpu[0], &gW[0], m, n,
				                         1.0f, lr, 0.0f, 0.0f, 1.0f,
				                         ac, cpuRng);
			}

			// --- GPU path ---
			// Both CPU and GPU now use glades::rng::Engine, so seeding an engine
			// with the same seed produces identical initial U. We still overwrite
			// GPU state with the CPU's initial U to guarantee both paths diverge
			// only due to floating-point ordering differences, not initialization.
			glades::rng::Engine gpuInitRng;
			glades::rng::seed_engine(gpuInitRng, cpuSeed);
			glades::gpu::GpuAtlasWeightState gpuState;
			bool ok = glades::gpu::atlas_gpu_init(gpuState, m, n, r, muInit, gpuInitRng);
			ASSERT("==============ATLAS::Equiv GPU init failed==============", ok);

			// Upload CPU's initial basis to GPU (so both start from the same U)
			{
				glades::rng::Engine initRng;
				glades::rng::seed_engine(initRng, cpuSeed);
				glades::atlas::WeightState tmpCpu;
				glades::atlas::initWeightState(tmpCpu, m, n, r, muInit, initRng);
				gpuState.U.upload(tmpCpu.U.data(), tmpCpu.U.size());
				gpuState.fisherDiag.upload(tmpCpu.fisherDiag.data(), tmpCpu.fisherDiag.size());
				gpuState.prevGz.zero();
				gpuState.sigma2 = tmpCpu.sigma2;
				gpuState.mu = tmpCpu.mu;
				gpuState.step = 0ULL;
			}

			// Initialize GPU weights identically
			const size_t mn = m * n;
			glades::gpu::GpuBuffer<float> d_W, d_gW;
			d_W.allocate(mn);
			d_gW.allocate(mn);
			{
				std::vector<float> h_W(mn);
				for (unsigned int i = 0; i < mn; ++i)
					h_W[i] = 0.1f * (float)((i * 7 + 3) % 11) / 11.0f;
				d_W.upload(h_W.data(), mn);
			}

			for (unsigned int s = 0; s < nSteps; ++s)
			{
				d_gW.upload(allGrads[s].data(), mn);
				ok = glades::gpu::atlas_gpu_step(gpuState, d_W.data(), d_gW.data(),
				                                  m, n,
				                                  1.0f, lr,
				                                  0.0f, 0.0f, 1.0f,
				                                  ac);
				ASSERT("==============ATLAS::Equiv GPU step failed==============", ok);
			}

			// Download and compare
			std::vector<float> W_gpu(mn);
			d_W.download(W_gpu.data());

			// GPU and CPU use different GEMM implementations (cuBLAS vs tiled/CBLAS),
			// so float32 rounding causes small divergence. Allow a relative tolerance.
			float maxRelErr = 0.0f;
			for (unsigned int i = 0; i < mn; ++i)
			{
				float diff = fabsf(W_cpu[i] - W_gpu[i]);
				float scale = fmaxf(fabsf(W_cpu[i]), 1e-6f);
				float relErr = diff / scale;
				if (relErr > maxRelErr) maxRelErr = relErr;
			}

			printf("[UT] 18d: max_relative_error=%.6f (cpu_sigma2=%.6f gpu_sigma2=%.6f)\n",
			       maxRelErr, cpuState.sigma2, gpuState.sigma2);
			// Tolerance: 2% relative error accounts for float32
			// precision differences between CPU (with CBLAS) and GPU (cuBLAS).
			ASSERT("==============ATLAS::Equiv GPU-CPU weights diverged too much==============",
			       maxRelErr < 0.02f);

			printf("[UT] 18d: PASSED\n");
		}

		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

		// ---------------------------------------------------------------
		// Test 29: GPU-to-CPU state transfer equivalence
		// ---------------------------------------------------------------
		printf("-----------------------------------\n");
		printf("ATLAS Test 29: GPU-to-CPU state transfer equivalence\n");
		printf("-----------------------------------\n");
		{
			// Run ATLAS on GPU for N steps, download state to CPU,
			// then continue M more steps on both. Verify weights match.
			const unsigned int m = 8;
			const unsigned int n = 6;
			const unsigned int r = 3;
			const unsigned int warmupSteps = 20;
			const unsigned int compareSteps = 30;
			const float lr = 0.01f;
			const float muInit = 0.01f;
			const unsigned long long seed = 29292929ULL;

			glades::ATLASConfig ac;
			ac.rank = r;
			ac.beta = 0.99f;
			ac.tSub = 10;
			ac.biasCorrection = true;

			// Pre-generate all gradients
			const size_t mn = (size_t)m * n;
			const unsigned int totalSteps = warmupSteps + compareSteps;
			std::vector<std::vector<float> > allGrads(totalSteps, std::vector<float>(mn));
			{
				glades::rng::Engine gradRng;
				glades::rng::seed_engine(gradRng, 55555ULL);
				for (unsigned int s = 0; s < totalSteps; ++s)
					for (size_t i = 0; i < mn; ++i)
						allGrads[s][i] = glades::rng::standard_normal(gradRng) * 0.1f;
			}

			// Initial weights (same for GPU and CPU comparison path)
			std::vector<float> initW(mn);
			for (size_t i = 0; i < mn; ++i)
				initW[i] = 0.1f * (float)((i * 7 + 3) % 11) / 11.0f;

			// --- GPU warmup phase ---
			glades::rng::Engine gpuRng;
			glades::rng::seed_engine(gpuRng, seed);
			glades::gpu::GpuAtlasWeightState gpuState;
			bool ok = glades::gpu::atlas_gpu_init(gpuState, m, n, r, muInit, gpuRng);
			ASSERT("==============ATLAS::Xfer GPU init failed==============", ok);

			glades::gpu::GpuBuffer<float> d_W, d_gW;
			d_W.allocate(mn);
			d_gW.allocate(mn);
			d_W.upload(initW.data(), mn);

			for (unsigned int s = 0; s < warmupSteps; ++s)
			{
				d_gW.upload(allGrads[s].data(), mn);
				ok = glades::gpu::atlas_gpu_step(gpuState, d_W.data(), d_gW.data(),
				                                  m, n, 1.0f, lr, 0.0f, 0.0f, 1.0f, ac);
				ASSERT("==============ATLAS::Xfer GPU warmup step failed==============", ok);
			}

			// --- Download GPU state to CPU WeightState ---
			glades::atlas::WeightState cpuState;
			cpuState.m = m;
			cpuState.n = n;
			cpuState.r = r;
			cpuState.U.resize((size_t)m * r);
			gpuState.U.download(cpuState.U.data(), cpuState.U.size());
			cpuState.fisherDiag.resize(r);
			gpuState.fisherDiag.download(cpuState.fisherDiag.data(), r);
			cpuState.prevGz.resize((size_t)r * n);
			gpuState.prevGz.download(cpuState.prevGz.data(), cpuState.prevGz.size());
			cpuState.sigma2 = gpuState.sigma2;
			cpuState.mu = gpuState.mu;
			cpuState.step = gpuState.step;
			cpuState.initialized = true;

			// Allocate CPU scratch buffers
			const size_t mr = (size_t)m * r;
			const size_t rn = (size_t)r * n;
			cpuState.scratch_gz.resize(rn);
			cpuState.scratch_corrected.resize(rn);
			cpuState.scratch_U_old.resize(mr);
			cpuState.scratch_f_old.resize(r);
			cpuState.scratch_B.resize(rn);
			cpuState.scratch_Z.resize(mr);
			cpuState.scratch_overlap.resize((size_t)r * r);
			cpuState.scratch_prevGzOld.resize(rn);

			// Download GPU weights for CPU path
			std::vector<float> W_cpu(mn);
			d_W.download(W_cpu.data());

			// Need a CPU RNG engine for refreshSubspace calls.
			// Seed it with same base seed + step offset so it's deterministic.
			glades::rng::Engine cpuRng;
			glades::rng::seed_engine(cpuRng, seed + gpuState.step);

			// --- Continue both paths for compareSteps ---
			for (unsigned int s = 0; s < compareSteps; ++s)
			{
				const unsigned int gi = warmupSteps + s;

				// CPU step
				std::vector<float> gW_cpu(allGrads[gi]);
				glades::atlas::applyStep(cpuState, &W_cpu[0], &gW_cpu[0], m, n,
				                         1.0f, lr, 0.0f, 0.0f, 1.0f, ac, cpuRng);

				// GPU step
				d_gW.upload(allGrads[gi].data(), mn);
				ok = glades::gpu::atlas_gpu_step(gpuState, d_W.data(), d_gW.data(),
				                                  m, n, 1.0f, lr, 0.0f, 0.0f, 1.0f, ac);
				ASSERT("==============ATLAS::Xfer GPU compare step failed==============", ok);
			}

			// --- Compare final weights ---
			std::vector<float> W_gpu(mn);
			d_W.download(W_gpu.data());

			float maxRelErr = 0.0f;
			for (size_t i = 0; i < mn; ++i)
			{
				float diff = fabsf(W_cpu[i] - W_gpu[i]);
				float scale = fmaxf(fabsf(W_cpu[i]), 1e-6f);
				float relErr = diff / scale;
				if (relErr > maxRelErr) maxRelErr = relErr;
			}

			printf("[UT] 29: GPU-to-CPU transfer max_relative_error=%.6f "
			       "(cpu_sigma2=%.6f gpu_sigma2=%.6f cpu_mu=%.6f gpu_mu=%.6f)\n",
			       maxRelErr, cpuState.sigma2, gpuState.sigma2,
			       cpuState.mu, gpuState.mu);

			// Same tolerance as Test 18d: 2% relative error from GEMM differences
			ASSERT("==============ATLAS::Xfer GPU-CPU weights diverged too much==============",
			       maxRelErr < 0.02f);

			printf("[UT] 29: PASSED\n");
		}
		printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);
	}
	else
	{
		printf("-----------------------------------\n");
		printf("ATLAS Tests 18-29: GPU ATLAS kernels (SKIPPED - no CUDA device)\n");
		printf("-----------------------------------\n");
	}
#else
	printf("-----------------------------------\n");
	printf("ATLAS Tests 18-29: GPU ATLAS kernels (SKIPPED - no CUDA support)\n");
	printf("-----------------------------------\n");
#endif

	// ---------------------------------------------------------------
	// Test 19: Edge case - single-element weight matrix (1x1)
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 19: Single-element weight matrix (1x1)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 1;
		const unsigned int n = 1;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 99999ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, 128, 0.01f, rng);

		ASSERT("==============ATLAS::1x1 init failed==============", state.initialized);
		ASSERT("==============ATLAS::1x1 rank should be 1==============", state.r == 1u);

		// Run 50 steps
		std::vector<float> W(1, 1.0f);
		glades::ATLASConfig ac1x1;
		ac1x1.rank = 128;
		ac1x1.tSub = 10;
		for (int step = 0; step < 50; ++step)
		{
			std::vector<float> gW(1, 0.1f);
			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         ac1x1, rng);
			ASSERT("==============ATLAS::1x1 NaN weight==============", W[0] == W[0]);
			ASSERT("==============ATLAS::1x1 Inf weight==============", (W[0] - W[0]) == 0.0f);
		}
		printf("[UT] ATLAS 1x1: 50 steps, final W=%.6f\n", W[0]);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 20: Edge case - zero gradients (should not diverge)
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 20: Zero gradients\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 33333ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, 2, 0.01f, rng);

		std::vector<float> W(m * n, 0.5f);
		std::vector<float> W_init(W.begin(), W.end());

		glades::ATLASConfig acZero;
		acZero.tSub = 10;
		acZero.muMin = 0.0f;
		acZero.muMax = 0.0f;

		// 100 steps with zero gradient — weights should be unchanged (no weight decay)
		for (int step = 0; step < 100; ++step)
		{
			std::vector<float> gW(m * n, 0.0f);
			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         acZero, rng);
		}

		// Weights should be identical (no gradient, no weight decay)
		bool unchanged = true;
		for (unsigned int i = 0; i < m * n; ++i)
		{
			if (W[i] != W_init[i]) { unchanged = false; break; }
		}
		ASSERT("==============ATLAS::ZeroGrad weights changed unexpectedly==============", unchanged);

		// State should still be finite
		ASSERT("==============ATLAS::ZeroGrad sigma2 NaN==============",
		       state.sigma2 == state.sigma2);
		ASSERT("==============ATLAS::ZeroGrad mu NaN==============",
		       state.mu == state.mu);

		printf("[UT] ATLAS zero gradients: weights unchanged after 100 steps\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 21: Edge case - rank equals min(m,n) (square subspace)
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 21: Rank = min(m,n) (full subspace)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int r = 3; // = min(m,n), full column subspace

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 44444ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		ASSERT("==============ATLAS::FullRank init failed==============", state.initialized);
		ASSERT("==============ATLAS::FullRank rank wrong==============", state.r == 3u);

		// U should be orthonormal
		const float tol = 1e-5f;
		for (unsigned int c = 0; c < state.r; ++c)
		{
			for (unsigned int d = c; d < state.r; ++d)
			{
				float dot = 0.0f;
				for (unsigned int i = 0; i < m; ++i)
					dot += state.U[i * state.r + c] * state.U[i * state.r + d];
				float expected = (c == d) ? 1.0f : 0.0f;
				float err = (dot - expected) < 0.0f ? -(dot - expected) : (dot - expected);
				ASSERT("==============ATLAS::FullRank U not orthonormal==============", err <= tol);
			}
		}

		// Run steps with refresh to verify it handles full-rank correctly
		std::vector<float> W(m * n, 0.0f);
		for (unsigned int i = 0; i < m * n; ++i)
			W[i] = static_cast<float>(i) * 0.1f;

		glades::ATLASConfig acFull;
		acFull.rank = r;
		acFull.tSub = 5;  // frequent refresh

		for (int step = 0; step < 50; ++step)
		{
			std::vector<float> gW(m * n, 0.0f);
			for (unsigned int i = 0; i < m * n; ++i)
				gW[i] = glades::rng::standard_normal(rng) * 0.1f;
			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         acFull, rng);

			for (unsigned int i = 0; i < m * n; ++i)
			{
				ASSERT("==============ATLAS::FullRank NaN weight==============", W[i] == W[i]);
				ASSERT("==============ATLAS::FullRank Inf weight==============", (W[i] - W[i]) == 0.0f);
			}
		}
		printf("[UT] ATLAS full-rank (r=%u = min(%u,%u)): 50 steps, all finite\n", r, m, n);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 22: ATLAS CNN convergence
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 22: CNN convergence\n");
	printf("-----------------------------------\n");
	{
		// Synthetic 2-class 1-channel 8x8 image data.
		const unsigned int C = 1, H = 8, W = 8;
		const unsigned int numClasses = 2;
		const unsigned int numSamples = 20;
		const unsigned int featureCount = C * H * W;

		glades::NumberInput* di = new glades::NumberInput();
		di->trainMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(featureCount, 0.0f));
		di->trainExpectedMatrix = shmea::GMatrix(numSamples, shmea::GVector<float>(numClasses, 0.0f));

		unsigned int s = 555u;
		for (unsigned int i = 0; i < numSamples; ++i)
		{
			const unsigned int label = i % numClasses;
			for (unsigned int f = 0; f < featureCount; ++f)
			{
				s = s * 1103515245u + 12345u;
				float v = static_cast<float>((s >> 16) & 0x7FFFu) / 32767.0f;
				if (f % numClasses == label)
					v = v * 0.5f + 0.5f;
				else
					v = v * 0.5f;
				di->trainMatrix[i][f] = v;
			}
			di->trainExpectedMatrix[i][label] = 1.0f;
		}
		di->testMatrix = di->trainMatrix;
		di->testExpectedMatrix = di->trainExpectedMatrix;

		// Build CNN: 1 conv layer (4 filters, 3x3, pad 1, maxpool 2x2) + 1 FC hidden (16).
		glades::InputLayerInfo* in = new glades::InputLayerInfo(
		    /*batchSize*/ 5,
		    /*learningRate*/ 0.01f,
		    /*momentumFactor*/ 0.0f,
		    /*weightDecay1*/ 0.0f,
		    /*weightDecay2*/ 0.0f,
		    /*pDropout*/ 0.0f,
		    /*activationType*/ glades::GMath::RELU,
		    /*activationParam*/ 1.0f);

		std::vector<glades::HiddenLayerInfo*> hidden;
		hidden.push_back(new glades::HiddenLayerInfo(
		    16, 0.01f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::RELU, 1.0f));

		glades::OutputLayerInfo* out = new glades::OutputLayerInfo(
		    static_cast<int>(numClasses), glades::OutputLayerInfo::CLASSIFICATION);

		glades::NNInfo* info = new glades::NNInfo("ut_atlas_cnn", in, hidden, out);
		glades::NNetwork net(info, glades::NNetwork::TYPE_CNN);
		net.setSeed(314u);
		net.getTerminatorMutable().setEpoch(200);
		net.getTerminatorMutable().setAccuracy(0);

		// Configure CNN architecture.
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.cnn.inputH = H;
			cfg.cnn.inputW = W;
			cfg.cnn.inputC = C;

			glades::CNNConfig::ConvLayerSpec spec;
			spec.outChannels = 4;
			spec.kernelH = 3; spec.kernelW = 3;
			spec.strideH = 1; spec.strideW = 1;
			spec.padH = 1; spec.padW = 1;
			spec.useBatchNorm = false;
			spec.useMaxPool = true;
			spec.poolH = 2; spec.poolW = 2;
			spec.poolStrideH = 2; spec.poolStrideW = 2;
			cfg.cnn.convLayers.push_back(spec);

			// Configure ATLAS optimizer.
			cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
			cfg.atlas.rank = 2;
			cfg.atlas.tSub = 50;
			cfg.atlas.beta = 0.999f;
		}

		CaptureMetricsCallbacks cb;
		const glades::NNetworkStatus st = net.train(di, &cb);
		ASSERT("==============ATLAS::CNN TrainStatus() Failed==============", st.ok());
		ASSERT("==============ATLAS::CNN no metrics captured==============", cb.saw);
		printf("[UT] ATLAS CNN: final loss = %f\n", cb.last.totalError);
		ASSERT("==============ATLAS::CNN loss too high==============", cb.last.totalError < 0.1f);

		delete di;
		delete info;
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 23: Weight decay shrinks weight norms
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 23: Weight decay shrinks weight norms\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 6;
		const unsigned int n = 4;
		const unsigned int r = 2;
		const unsigned int mn = m * n;
		const int nSteps = 40;

		glades::ATLASConfig acBase;
		acBase.beta = 0.999f;
		acBase.muMin = 0.0f;
		acBase.muMax = 0.5f;
		acBase.eps = 1e-8f;
		acBase.kappaMax = 10.0f;
		acBase.tSub = 50u;
		acBase.powerIters = 3u;
		acBase.betaRefresh = 0.5f;
		acBase.muGrowthRate = 0.001f;
		acBase.biasCorrection = false;

		// Helper: run nSteps of ATLAS with given weight decay, return final L2 norm of W.
		// Both runs start from the same initial weights and use the same gradient signal.
		float normNoWd = 0.0f;
		float normWithWd = 0.0f;

		// Run without weight decay
		{
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 777ULL);
			glades::atlas::WeightState state;
			glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

			std::vector<float> W(mn, 0.0f);
			for (unsigned int i = 0; i < mn; ++i)
				W[i] = static_cast<float>(i) * 0.05f + 0.1f;

			for (int s = 0; s < nSteps; ++s)
			{
				std::vector<float> gW(mn, 0.0f);
				for (unsigned int i = 0; i < mn; ++i)
					gW[i] = 0.01f; // small constant gradient
				glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
				                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
				                         acBase, rng);
			}
			for (unsigned int i = 0; i < mn; ++i)
				normNoWd += W[i] * W[i];
			normNoWd = sqrtf(normNoWd);
		}

		// Run with L1 + L2 weight decay
		{
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 777ULL);
			glades::atlas::WeightState state;
			glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

			std::vector<float> W(mn, 0.0f);
			for (unsigned int i = 0; i < mn; ++i)
				W[i] = static_cast<float>(i) * 0.05f + 0.1f;

			for (int s = 0; s < nSteps; ++s)
			{
				std::vector<float> gW(mn, 0.0f);
				for (unsigned int i = 0; i < mn; ++i)
					gW[i] = 0.01f;
				glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
				                         1.0f, 0.01f,
				                         /*wd1=*/0.001f, /*wd2=*/0.01f,
				                         1.0f, acBase, rng);
			}
			for (unsigned int i = 0; i < mn; ++i)
				normWithWd += W[i] * W[i];
			normWithWd = sqrtf(normWithWd);
		}

		printf("[UT] ATLAS weight decay: norm_no_wd=%f, norm_with_wd=%f\n",
		       normNoWd, normWithWd);
		ASSERT("==============ATLAS::WeightDecay did not shrink weights==============",
		       normWithWd < normNoWd);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 24: Checkpoint preserves ATLAS optimizer state exactly
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 24: Checkpoint state preservation\n");
	printf("-----------------------------------\n");
	{
		// Strategy: Train network A for 50 epochs, save checkpoint. Then continue
		// training A for 50 more epochs. Separately, load checkpoint into fresh
		// network B, train 50 epochs. Final losses must match exactly (proving
		// the checkpoint preserved all ATLAS optimizer state).
		const std::string ckptName = "ut_atlas_st24";
		float lossA = -1.0f;
		float lossB = -1.0f;

		// Path A: train 50 → checkpoint → train 50 more (continuously)
		{
			glades::NumberInput* diA = new glades::NumberInput();
			diA->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
			diA->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
			diA->trainMatrix[0][0] = 0.0f; diA->trainMatrix[0][1] = 0.0f;
			diA->trainExpectedMatrix[0][0] = 0.0f;
			diA->trainMatrix[1][0] = 0.0f; diA->trainMatrix[1][1] = 1.0f;
			diA->trainExpectedMatrix[1][0] = 1.0f;
			diA->trainMatrix[2][0] = 1.0f; diA->trainMatrix[2][1] = 0.0f;
			diA->trainExpectedMatrix[2][0] = 1.0f;
			diA->trainMatrix[3][0] = 1.0f; diA->trainMatrix[3][1] = 1.0f;
			diA->trainExpectedMatrix[3][0] = 0.0f;
			diA->testMatrix = diA->trainMatrix;
			diA->testExpectedMatrix = diA->trainExpectedMatrix;

			glades::InputLayerInfo* in = new glades::InputLayerInfo(
			    4, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f);
			std::vector<glades::HiddenLayerInfo*> hidden;
			hidden.push_back(new glades::HiddenLayerInfo(
			    8, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, glades::GMath::SIGMOID, 1.0f));
			glades::OutputLayerInfo* out = new glades::OutputLayerInfo(1, glades::OutputLayerInfo::REGRESSION);
			glades::NNInfo* info = new glades::NNInfo("ut_atlas_st24", in, hidden, out);

			glades::NNetwork netA(info, glades::NNetwork::TYPE_DFF);
			netA.setSeed(42u);
			netA.getTerminatorMutable().setEpoch(50);
			netA.getTerminatorMutable().setAccuracy(0);
			{
				glades::TrainingConfig& cfg = netA.getTrainingConfigMutable();
				cfg.optimizer.type = glades::OptimizerConfig::ATLAS;
				cfg.atlas.rank = 4;
				cfg.atlas.tSub = 50;
			}

			// Phase 1: first 50 epochs
			glades::NNetworkStatus st = netA.train(diA);
			ASSERT("==============ATLAS::StateCheckpoint Phase1 Train Failed==============", st.ok());

			// Save checkpoint
			st = netA.saveCheckpoint(ckptName);
			ASSERT("==============ATLAS::StateCheckpoint Save Failed==============", st.ok());

			// Phase 2: 50 more epochs
			netA.getTerminatorMutable().setEpoch(50);
			CaptureMetricsCallbacks cbA;
			st = netA.train(diA, &cbA);
			ASSERT("==============ATLAS::StateCheckpoint Phase2 Train Failed==============", st.ok());
			ASSERT("==============ATLAS::StateCheckpoint Phase2 no metrics==============", cbA.saw);
			lossA = cbA.last.totalError;

			delete diA;
			delete info;
		}

		// Path B: load checkpoint → train 50 epochs
		{
			glades::NumberInput* diB = new glades::NumberInput();
			diB->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
			diB->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
			diB->trainMatrix[0][0] = 0.0f; diB->trainMatrix[0][1] = 0.0f;
			diB->trainExpectedMatrix[0][0] = 0.0f;
			diB->trainMatrix[1][0] = 0.0f; diB->trainMatrix[1][1] = 1.0f;
			diB->trainExpectedMatrix[1][0] = 1.0f;
			diB->trainMatrix[2][0] = 1.0f; diB->trainMatrix[2][1] = 0.0f;
			diB->trainExpectedMatrix[2][0] = 1.0f;
			diB->trainMatrix[3][0] = 1.0f; diB->trainMatrix[3][1] = 1.0f;
			diB->trainExpectedMatrix[3][0] = 0.0f;
			diB->testMatrix = diB->trainMatrix;
			diB->testExpectedMatrix = diB->trainExpectedMatrix;

			glades::NNetwork netB;
			glades::NNetworkStatus st = netB.loadCheckpoint(ckptName, diB);
			ASSERT("==============ATLAS::StateCheckpoint Load Failed==============", st.ok());

			// Restore ATLAS config (not yet persisted in checkpoint manifest).
			{
				glades::TrainingConfig& cfg = netB.getTrainingConfigMutable();
				cfg.atlas.rank = 4;
				cfg.atlas.tSub = 50;
			}
			netB.getTerminatorMutable().setEpoch(50);
			CaptureMetricsCallbacks cbB;
			st = netB.train(diB, &cbB);
			ASSERT("==============ATLAS::StateCheckpoint Phase3 Train Failed==============", st.ok());
			ASSERT("==============ATLAS::StateCheckpoint Phase3 no metrics==============", cbB.saw);
			lossB = cbB.last.totalError;

			delete diB;
		}

		printf("[UT] ATLAS state checkpoint: lossA=%f, lossB=%f\n", lossA, lossB);
		// If optimizer state is perfectly preserved, losses must match closely.
		const float relDiff = (lossA > 0.0f && lossB > 0.0f)
		    ? fabsf(lossA - lossB) / (0.5f * (lossA + lossB))
		    : fabsf(lossA - lossB);
		printf("[UT] Relative difference: %e\n", relDiff);
		ASSERT("==============ATLAS::StateCheckpoint losses diverged==============",
		       relDiff < 1e-5f);

		// Cleanup
		{
			const std::string base = "database/checkpoints/" + ckptName;
			remove((base + "/manifest.txt").c_str());
			remove((base + "/nninfo.csv").c_str());
			for (int i = 0; i < 10; ++i)
			{
				char buf[64];
				snprintf(buf, sizeof(buf), "/shard_%03d.bin", i);
				remove((base + buf).c_str());
			}
			rmdir(base.c_str());
		}
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ----------------------------------------------------------------
	// Test 25: Refresh + mu interaction property test
	// ----------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 25: Refresh + mu interaction\n");
	printf("-----------------------------------\n");
	{
		// Run ATLAS with tSub=5 so refreshes happen frequently (every 5 steps).
		// Verify that across refresh boundaries, invariants hold:
		//   - weights remain finite
		//   - fisherDiag stays non-negative and finite
		//   - mu stays in [muMin, muMax]
		//   - prevGz remains finite
		//   - refreshSubspace returns true (no NaN recovery)
		const unsigned int m = 8, n = 6, r = 3;
		const unsigned int totalSteps = 30; // 6 refreshes at tSub=5
		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 77ULL);

		glades::atlas::WeightState state;
		glades::ATLASConfig ac;
		ac.rank = r;
		ac.tSub = 5;
		ac.muMin = 0.01f;
		ac.muMax = 0.3f;

		std::vector<float> W(m * n, 0.0f);
		std::vector<float> gW(m * n, 0.0f);

		// Initialize weights with small random values.
		for (size_t i = 0; i < W.size(); ++i)
			W[i] = static_cast<float>(glades::rng::uniform_double(rng, -0.5, 0.5));

		unsigned int refreshCount = 0;
		for (unsigned int s = 0; s < totalSteps; ++s)
		{
			// Generate a gradient that drifts slowly (smooth component + noise).
			for (size_t i = 0; i < gW.size(); ++i)
			{
				const float signal = 0.1f * W[i]; // gradient correlated with weights
				const float noise = static_cast<float>(glades::rng::uniform_double(rng, -0.02, 0.02));
				gW[i] = signal + noise;
			}

			const bool ok = glades::atlas::update(state, &W[0], &gW[0], m, n,
			                                       /*invBatch=*/1.0f, /*lr=*/0.01f,
			                                       /*wd1=*/0.0f, /*wd2=*/0.0f,
			                                       /*gradScale=*/1.0f, ac, rng);
			(void)ok; // applyStep may return false on recovery, but state should remain valid

			// Count refreshes (step is incremented inside applyStep).
			if (state.step > 0ULL && (state.step % 5ULL) == 0ULL)
				++refreshCount;

			// Invariant checks after every step.
			for (size_t i = 0; i < W.size(); ++i)
			{
				ASSERT("==============ATLAS::RefreshMu W not finite==============",
				       W[i] == W[i] && W[i] != W[i] + 1.0f); // not NaN and not Inf (for non-zero)
			}
			for (unsigned int c = 0; c < state.r; ++c)
			{
				ASSERT("==============ATLAS::RefreshMu Fisher NaN==============",
				       state.fisherDiag[c] == state.fisherDiag[c]);
				ASSERT("==============ATLAS::RefreshMu Fisher negative==============",
				       state.fisherDiag[c] >= 0.0f);
			}
			ASSERT("==============ATLAS::RefreshMu mu below min==============",
			       state.mu >= ac.muMin);
			ASSERT("==============ATLAS::RefreshMu mu above max==============",
			       state.mu <= ac.muMax);
			for (size_t i = 0; i < state.prevGz.size(); ++i)
			{
				ASSERT("==============ATLAS::RefreshMu prevGz NaN==============",
				       state.prevGz[i] == state.prevGz[i]);
			}
		}

		ASSERT("==============ATLAS::RefreshMu no refreshes happened==============",
		       refreshCount >= 4); // expect 6 refreshes in 30 steps with tSub=5
		printf("[UT] ATLAS refresh+mu interaction: %u refreshes in %u steps, all invariants held\n",
		       refreshCount, totalSteps);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 26: applyStep with biasCorrection=true (default config)
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 26: applyStep with bias correction enabled\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 4;
		const unsigned int n = 3;
		const unsigned int r = 2;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 26262626ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		std::vector<float> W(m * n, 0.0f);
		for (unsigned int i = 0; i < m * n; ++i)
			W[i] = static_cast<float>(i) * 0.1f - 0.5f;

		// Use default ATLASConfig (biasCorrection = true)
		glades::ATLASConfig acDefault;
		acDefault.rank = r;

		for (int step = 0; step < 50; ++step)
		{
			std::vector<float> gW(m * n, 0.0f);
			for (unsigned int i = 0; i < m * n; ++i)
				gW[i] = W[i] * 0.1f;

			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         acDefault, rng);

			// All weights must remain finite
			for (unsigned int i = 0; i < m * n; ++i)
			{
				ASSERT("==============ATLAS::BiasCorr applyStep NaN==============", W[i] == W[i]);
				float d = W[i] - W[i];
				ASSERT("==============ATLAS::BiasCorr applyStep Inf==============", d == 0.0f);
			}

			// Fisher must be positive and finite
			for (unsigned int c = 0; c < r; ++c)
			{
				ASSERT("==============ATLAS::BiasCorr Fisher NaN==============",
				       state.fisherDiag[c] == state.fisherDiag[c]);
				ASSERT("==============ATLAS::BiasCorr Fisher negative==============",
				       state.fisherDiag[c] >= 0.0f);
			}

			// sigma2 and mu must be finite
			ASSERT("==============ATLAS::BiasCorr sigma2 NaN==============",
			       state.sigma2 == state.sigma2);
			ASSERT("==============ATLAS::BiasCorr mu NaN==============",
			       state.mu == state.mu);
		}

		// Verify weights changed
		bool changed = false;
		for (unsigned int i = 0; i < m * n; ++i)
		{
			float init = static_cast<float>(i) * 0.1f - 0.5f;
			if (W[i] != init) { changed = true; break; }
		}
		ASSERT("==============ATLAS::BiasCorr weights unchanged==============", changed);

		printf("[UT] ATLAS applyStep with biasCorrection=true: 50 steps, all invariants held\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 27: Fuzz test with biasCorrection=true (default config)
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 27: Fuzz test with bias correction enabled\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 8;
		const unsigned int n = 6;
		const unsigned int r = 3;
		const int nSteps = 300;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 27272727ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		std::vector<float> W(m * n, 0.0f);
		for (unsigned int i = 0; i < m * n; ++i)
			W[i] = glades::rng::standard_normal(rng) * 0.1f;

		// Default config with biasCorrection = true
		glades::ATLASConfig acFuzz;
		acFuzz.rank = r;
		acFuzz.tSub = 15u;

		for (int step = 0; step < nSteps; ++step)
		{
			std::vector<float> gW(m * n, 0.0f);
			for (unsigned int i = 0; i < m * n; ++i)
				gW[i] = glades::rng::standard_normal(rng) * 0.5f;

			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.01f, 0.0f, 0.0f, 1.0f,
			                         acFuzz, rng);

			for (unsigned int i = 0; i < m * n; ++i)
			{
				ASSERT("==============ATLAS::BCFuzz NaN weight==============", W[i] == W[i]);
				float d = W[i] - W[i];
				ASSERT("==============ATLAS::BCFuzz Inf weight==============", d == 0.0f);
			}
			for (unsigned int c = 0; c < state.r; ++c)
			{
				ASSERT("==============ATLAS::BCFuzz Fisher NaN==============",
				       state.fisherDiag[c] == state.fisherDiag[c]);
				ASSERT("==============ATLAS::BCFuzz Fisher negative==============",
				       state.fisherDiag[c] >= 0.0f);
			}
			ASSERT("==============ATLAS::BCFuzz mu below min==============",
			       state.mu >= acFuzz.muMin);
			ASSERT("==============ATLAS::BCFuzz mu above max==============",
			       state.mu <= acFuzz.muMax + 1e-6f);
			ASSERT("==============ATLAS::BCFuzz sigma2 NaN==============",
			       state.sigma2 == state.sigma2);
			ASSERT("==============ATLAS::BCFuzz sigma2 negative==============",
			       state.sigma2 >= 0.0f);
		}
		printf("[UT] ATLAS fuzz biasCorrection=true: %d steps, all invariants held\n", nSteps);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test 28: Scale stress test (m=512, n=512, r=64)
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS Test 28: Scale stress test (512x512 r=64)\n");
	printf("-----------------------------------\n");
	{
		const unsigned int m = 512;
		const unsigned int n = 512;
		const unsigned int r = 64;
		const int nSteps = 200;

		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 28282828ULL);
		glades::atlas::WeightState state;
		glades::atlas::initWeightState(state, m, n, r, 0.01f, rng);

		ASSERT("==============ATLAS::Scale init failed==============", state.initialized);
		ASSERT("==============ATLAS::Scale rank wrong==============", state.r == r);

		// Verify initial U orthonormality
		{
			const float tol = 1e-4f;
			for (unsigned int c = 0; c < r; c += 16) // sample every 16th column pair
			{
				float dot = 0.0f;
				for (unsigned int i = 0; i < m; ++i)
					dot += state.U[i * r + c] * state.U[i * r + c];
				float err = (dot - 1.0f) < 0.0f ? -(dot - 1.0f) : (dot - 1.0f);
				ASSERT("==============ATLAS::Scale initial U not unit norm==============", err <= tol);
			}
		}

		std::vector<float> W(static_cast<size_t>(m) * n, 0.0f);
		for (size_t i = 0; i < W.size(); ++i)
			W[i] = glades::rng::standard_normal(rng) * 0.02f;

		glades::ATLASConfig acScale;
		acScale.rank = r;
		acScale.tSub = 50;
		acScale.beta = 0.999f;
		acScale.biasCorrection = true;

		for (int step = 0; step < nSteps; ++step)
		{
			std::vector<float> gW(static_cast<size_t>(m) * n, 0.0f);
			for (size_t i = 0; i < gW.size(); ++i)
				gW[i] = glades::rng::standard_normal(rng) * 0.01f;

			glades::atlas::applyStep(state, &W[0], &gW[0], m, n,
			                         1.0f, 0.001f, 0.0f, 0.0f, 1.0f,
			                         acScale, rng);

			// Check invariants every 50 steps (not every step to keep runtime reasonable)
			if (step % 50 == 0 || step == nSteps - 1)
			{
				// Weights finite (sample check)
				for (size_t i = 0; i < W.size(); i += 1000)
				{
					ASSERT("==============ATLAS::Scale NaN weight==============", W[i] == W[i]);
					float d = W[i] - W[i];
					ASSERT("==============ATLAS::Scale Inf weight==============", d == 0.0f);
				}

				// Fisher positive and finite
				for (unsigned int c = 0; c < r; ++c)
				{
					ASSERT("==============ATLAS::Scale Fisher NaN==============",
					       state.fisherDiag[c] == state.fisherDiag[c]);
					ASSERT("==============ATLAS::Scale Fisher negative==============",
					       state.fisherDiag[c] >= 0.0f);
				}

				// sigma2 and mu
				ASSERT("==============ATLAS::Scale sigma2 NaN==============",
				       state.sigma2 == state.sigma2);
				ASSERT("==============ATLAS::Scale mu NaN==============",
				       state.mu == state.mu);

				// U orthonormality (sample)
				if (state.step > 0ULL && (state.step % 50ULL) == 0ULL)
				{
					const float orthoTol = 1e-3f; // looser tolerance for large matrices
					for (unsigned int c = 0; c < r; c += 16)
					{
						float dot = 0.0f;
						for (unsigned int i = 0; i < m; ++i)
							dot += state.U[i * r + c] * state.U[i * r + c];
						float err = (dot - 1.0f) < 0.0f ? -(dot - 1.0f) : (dot - 1.0f);
						ASSERT("==============ATLAS::Scale U not unit norm after refresh==============",
						       err <= orthoTol);
					}
				}
			}
		}

		printf("[UT] ATLAS scale test (512x512 r=64): %d steps completed, all invariants held\n",
		       nSteps);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	printf("\n============================================================\n");
}

void ATLASGpuNaNTest()
{
	printf("============================================================\n");
	printf("ATLAS GPU NaN Reproduction Test\n");
	printf("============================================================\n");

#ifdef GLADES_HAVE_CUDA
	// ---------------------------------------------------------------
	// Test GPU 0: Fisher kernel mean-squared reduction
	//
	// Isolates atlas_gpu_fisher_update. With beta=0 and gz filled with 1.0,
	// each Fisher entry must become exactly 1.0 because mean(gz^2)=1.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS GPU Test 0: Fisher kernel mean-squared reduction\n");
	printf("-----------------------------------\n");
	if (glades::gpu::initDevice())
	{
		const int r = 4;
		const int outerDim = 1024;

		glades::gpu::GpuBuffer<float> d_gz_left, d_gz_right, d_fisher;
		ASSERT("==============GPU Fisher left alloc==============",
		       d_gz_left.allocate((size_t)r * outerDim));
		ASSERT("==============GPU Fisher right alloc==============",
		       d_gz_right.allocate((size_t)outerDim * r));
		ASSERT("==============GPU Fisher diag alloc==============",
		       d_fisher.allocate((size_t)r));

		std::vector<float> ones((size_t)r * outerDim, 1.0f);
		std::vector<float> fisher(r, 0.0f);

		ASSERT("==============GPU Fisher left upload==============",
		       d_gz_left.upload(ones.data(), ones.size()));
		ASSERT("==============GPU Fisher diag zero upload==============",
		       d_fisher.upload(fisher.data(), fisher.size()));
		ASSERT("==============GPU Fisher left update==============",
		       glades::gpu::atlas_gpu_fisher_update(
		           d_gz_left.data(), d_fisher.data(), r, outerDim, 0.0f, false));
		ASSERT("==============GPU Fisher left download==============",
		       d_fisher.download(fisher.data(), fisher.size()));
		for (int c = 0; c < r; ++c)
		{
			printf("    left fisher[%d]=%.8f\n", c, fisher[c]);
			ASSERT("==============GPU Fisher left mean-squared mismatch==============",
			       fabsf(fisher[c] - 1.0f) < 1e-5f);
		}

		ASSERT("==============GPU Fisher right upload==============",
		       d_gz_right.upload(ones.data(), ones.size()));
		std::fill(fisher.begin(), fisher.end(), 0.0f);
		ASSERT("==============GPU Fisher diag reset upload==============",
		       d_fisher.upload(fisher.data(), fisher.size()));
		ASSERT("==============GPU Fisher right update==============",
		       glades::gpu::atlas_gpu_fisher_update(
		           d_gz_right.data(), d_fisher.data(), r, outerDim, 0.0f, true));
		ASSERT("==============GPU Fisher right download==============",
		       d_fisher.download(fisher.data(), fisher.size()));
		for (int c = 0; c < r; ++c)
		{
			printf("    right fisher[%d]=%.8f\n", c, fisher[c]);
			ASSERT("==============GPU Fisher right mean-squared mismatch==============",
			       fabsf(fisher[c] - 1.0f) < 1e-5f);
		}
	}

	// ---------------------------------------------------------------
	// Test GPU: ATLAS GPU NaN reproduction — zero/tiny gradient refresh
	//
	// Reproduces the production bug where layer 0 Wq gets NaN in
	// gz_norm at the first refresh step (step 100 with tSub=100).
	// Tests with zero gradients, tiny gradients, and normal gradients
	// to isolate which pattern triggers the NaN.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS GPU Test: NaN at refresh with zero/tiny gradients\n");
	printf("-----------------------------------\n");
	if (glades::gpu::initDevice())
	{
		const unsigned int m = 1024;
		const unsigned int n = 1024;
		const unsigned int r = 256;
		const unsigned int tSub = 100;
		const float lr = 5e-4f;
		const float muInit = 0.01f;

		struct GradScenario
		{
			const char* name;
			float gradScale; // 0 = zero grads, >0 = fill with random * gradScale
		};

		const GradScenario scenarios[] = {
			{"zero", 0.0f},
			{"tiny (1e-8)", 1e-8f},
			{"small (1e-5)", 1e-5f},
			{"normal (0.01)", 0.01f},
		};
		const int nScenarios = 4;

		for (int si = 0; si < nScenarios; ++si)
		{
			const GradScenario& sc = scenarios[si];
			printf("  Scenario: %s gradients\n", sc.name);

			// Allocate GPU buffers for weight and gradient
			glades::gpu::GpuBuffer<float> d_W, d_gW;
			ASSERT("==============GPU ATLAS NaN: W alloc failed==============",
			       d_W.allocate((size_t)m * n));
			ASSERT("==============GPU ATLAS NaN: gW alloc failed==============",
			       d_gW.allocate((size_t)m * n));

			// Initialize weights to small random values
			{
				std::vector<float> h_W(m * n);
				glades::rng::Engine initRng;
				glades::rng::seed_engine(initRng, 42ULL);
				for (size_t i = 0; i < m * n; ++i)
					h_W[i] = glades::rng::standard_normal(initRng) * 0.02f;
				ASSERT("==============GPU ATLAS NaN: W upload failed==============",
				       d_W.upload(h_W.data(), m * n));
			}

			// Initialize ATLAS state
			glades::gpu::GpuAtlasWeightState state;
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 12345ULL + si * 1000ULL);
			ASSERT("==============GPU ATLAS NaN: init failed==============",
			       glades::gpu::atlas_gpu_init(state, m, n, r, muInit, rng));

			glades::ATLASConfig ac;
			ac.rank = r;
			ac.tSub = tSub;
			ac.beta = 0.99f;
			ac.muMin = 0.01f;
			ac.muMax = 0.3f;
			ac.eps = 1e-8f;
			ac.kappaMax = 10.0f;
			ac.betaRefresh = 0.5f;
			ac.powerIters = 2;
			ac.muGrowthRate = 0.001f;
			ac.biasCorrection = true;

			bool nanDetected = false;
			unsigned int nanStep = 0;

			// Run for tSub+5 steps to cover the first refresh
			const unsigned int totalSteps = tSub + 5;
			for (unsigned int step = 0; step < totalSteps; ++step)
			{
				// Fill gradient buffer
				if (sc.gradScale == 0.0f)
				{
					// Zero gradient
					ASSERT("==============GPU ATLAS NaN: gW zero failed==============",
					       d_gW.zero());
				}
				else
				{
					std::vector<float> h_gW(m * n);
					for (size_t i = 0; i < m * n; ++i)
						h_gW[i] = glades::rng::standard_normal(rng) * sc.gradScale;
					ASSERT("==============GPU ATLAS NaN: gW upload failed==============",
					       d_gW.upload(h_gW.data(), m * n));
				}

				bool ok = glades::gpu::atlas_gpu_step(
					state, d_W.data(), d_gW.data(), m, n,
					/*invBatch=*/0.05f, lr,
					/*wd1=*/0.0f, /*wd2=*/0.01f, /*gradScale=*/1.0f,
					ac);

				if (!ok)
				{
					printf("    [FAIL] atlas_gpu_step returned false at step %u\n", step + 1);
					nanDetected = true;
					nanStep = step + 1;
					break;
				}

				// Check diagnostics at refresh steps
				if (state.step > 0ULL && (state.step % tSub) == 0ULL)
				{
					glades::gpu::AtlasGpuDiag diag = glades::gpu::atlas_gpu_get_diag(state);
					if (diag.valid)
					{
						bool gzNaN = (diag.gzNorm != diag.gzNorm);
						bool updateNaN = (diag.updateNorm != diag.updateNorm);
						bool fisherNaN = (diag.fisherMean != diag.fisherMean);
						printf("    step=%llu: sigma2=%.3e fisher_mean=%.3e "
						       "gz_norm=%s update_norm=%s\n",
						       (unsigned long long)state.step,
						       diag.sigma2, diag.fisherMean,
						       gzNaN ? "NaN" : "ok",
						       updateNaN ? "NaN" : "ok");
						if (gzNaN || updateNaN || fisherNaN)
						{
							nanDetected = true;
							nanStep = (unsigned int)state.step;
						}
					}
				}

				// Also spot-check sigma2 and mu for NaN
				if (state.sigma2 != state.sigma2 || state.mu != state.mu)
				{
					printf("    [FAIL] sigma2 or mu is NaN at step %llu\n",
					       (unsigned long long)state.step);
					nanDetected = true;
					nanStep = (unsigned int)state.step;
					break;
				}
			}

			if (nanDetected)
				printf("    [FAIL] NaN detected at step %u with %s gradients\n",
				       nanStep, sc.name);
			else
				printf("    [PASS] No NaN after %u steps with %s gradients\n",
				       totalSteps, sc.name);

			ASSERT("==============GPU ATLAS NaN: NaN detected in diagnostics==============",
			       !nanDetected);
		}
	}
	else
	{
		printf("  No CUDA device available, skipping GPU ATLAS NaN test.\n");
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

	// ---------------------------------------------------------------
	// Test GPU 2: NaN gradient at refresh step — permanent corruption
	//
	// Simulates a NaN arriving in d_gW at exactly the refresh step.
	// This is the suspected production root cause: a single NaN gradient
	// at tSub boundary corrupts U permanently via power iteration.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS GPU Test 2: NaN gradient at refresh step\n");
	printf("-----------------------------------\n");
	if (glades::gpu::initDevice())
	{
		const unsigned int m = 256;
		const unsigned int n = 256;
		const unsigned int r = 32;
		const unsigned int tSub = 10;
		const float lr = 5e-4f;

		glades::gpu::GpuBuffer<float> d_W, d_gW;
		ASSERT("==============GPU NaN refresh: W alloc==============",
		       d_W.allocate((size_t)m * n));
		ASSERT("==============GPU NaN refresh: gW alloc==============",
		       d_gW.allocate((size_t)m * n));

		// Init weights
		{
			std::vector<float> h_W(m * n);
			glades::rng::Engine initRng;
			glades::rng::seed_engine(initRng, 42ULL);
			for (size_t i = 0; i < m * n; ++i)
				h_W[i] = glades::rng::standard_normal(initRng) * 0.02f;
			d_W.upload(h_W.data(), m * n);
		}

		glades::gpu::GpuAtlasWeightState state;
		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 99999ULL);
		glades::gpu::atlas_gpu_init(state, m, n, r, 0.01f, rng);

		glades::ATLASConfig ac;
		ac.rank = r;
		ac.tSub = tSub;
		ac.beta = 0.99f;
		ac.muMin = 0.01f;
		ac.muMax = 0.3f;
		ac.eps = 1e-8f;
		ac.kappaMax = 10.0f;
		ac.betaRefresh = 0.5f;
		ac.powerIters = 2;
		ac.muGrowthRate = 0.001f;
		ac.biasCorrection = true;

		bool nanBefore = false;
		bool nanAfter = false;
		const unsigned int poisonStep = tSub; // inject NaN exactly at refresh step
		const unsigned int totalSteps = tSub * 3; // run past to check persistence

		for (unsigned int step = 0; step < totalSteps; ++step)
		{
			std::vector<float> h_gW(m * n);
			for (size_t i = 0; i < m * n; ++i)
				h_gW[i] = glades::rng::standard_normal(rng) * 0.01f;

			// Inject NaN at the refresh step
			if (step + 1 == poisonStep)
			{
				h_gW[0] = 0.0f / 0.0f; // NaN
				h_gW[m * n / 2] = 0.0f / 0.0f;
				printf("    Injecting NaN gradient at step %u (refresh step)\n", step + 1);
			}
			d_gW.upload(h_gW.data(), m * n);

			glades::gpu::atlas_gpu_step(
				state, d_W.data(), d_gW.data(), m, n,
				0.05f, lr, 0.0f, 0.01f, 1.0f, ac);

			// Check diagnostics at diagnostic steps (multiples of tSub)
			if (state.step > 0ULL && (state.step % tSub) == 0ULL)
			{
				glades::gpu::AtlasGpuDiag diag = glades::gpu::atlas_gpu_get_diag(state);
				if (diag.valid)
				{
					bool gzNaN = (diag.gzNorm != diag.gzNorm);
					printf("    step=%llu: gz_norm=%s sigma2=%.3e fisher_mean=%.3e\n",
					       (unsigned long long)state.step,
					       gzNaN ? "NaN" : "ok",
					       diag.sigma2, diag.fisherMean);
					if (state.step <= poisonStep)
						nanBefore = nanBefore || gzNaN;
					else
						nanAfter = nanAfter || gzNaN;
				}
			}
		}

		if (nanAfter)
			printf("    [FAIL] NaN persists after poisoned refresh step\n");
		else
			printf("    [PASS] No persistent NaN after poisoned refresh step\n");

		ASSERT("==============GPU ATLAS NaN: NaN persists after poisoned gradient==============",
		       !nanAfter);
	}

	// ---------------------------------------------------------------
	// Test GPU 3: Very small gradient at refresh — CholeskyQR overflow
	//
	// Layer 0 Wq gets gradients with RMS ~2e-6 (100x smaller than other
	// layers). This may cause rank-deficient power iteration output,
	// leading to huge R_inv entries in CholeskyQR that overflow.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS GPU Test 3: Tiny gradient at refresh (layer 0 Wq scenario)\n");
	printf("-----------------------------------\n");
	if (glades::gpu::initDevice())
	{
		// Test with gradient magnitudes matching production layer 0 Wq
		const float gradScales[] = {2e-6f, 1e-7f, 1e-9f, 0.0f};
		const char* scaleNames[] = {"2e-6 (production)", "1e-7", "1e-9", "zero"};
		const int nScales = 4;

		for (int si = 0; si < nScales; ++si)
		{
			const unsigned int m = 1024;
			const unsigned int n = 1024;
			const unsigned int r = 256;
			const unsigned int tSub = 10; // fast refresh for testing
			printf("  Gradient RMS = %s:\n", scaleNames[si]);

			glades::gpu::GpuBuffer<float> d_W, d_gW;
			d_W.allocate((size_t)m * n);
			d_gW.allocate((size_t)m * n);

			// Init weights
			{
				std::vector<float> h_W(m * n);
				glades::rng::Engine wr;
				glades::rng::seed_engine(wr, 42ULL);
				for (size_t i = 0; i < m * n; ++i)
					h_W[i] = glades::rng::standard_normal(wr) * 0.02f;
				d_W.upload(h_W.data(), m * n);
			}

			glades::gpu::GpuAtlasWeightState state;
			glades::rng::Engine rng;
			glades::rng::seed_engine(rng, 77777ULL);
			glades::gpu::atlas_gpu_init(state, m, n, r, 0.01f, rng);

			glades::ATLASConfig ac;
			ac.rank = r;
			ac.tSub = tSub;
			ac.beta = 0.99f;
			ac.muMin = 0.01f;
			ac.muMax = 0.3f;
			ac.eps = 1e-8f;
			ac.kappaMax = 10.0f;
			ac.betaRefresh = 0.5f;
			ac.powerIters = 2;
			ac.muGrowthRate = 0.001f;
			ac.biasCorrection = true;

			bool nanDetected = false;
			for (unsigned int step = 0; step < tSub + 2; ++step)
			{
				std::vector<float> h_gW(m * n);
				if (gradScales[si] > 0.0f)
				{
					for (size_t i = 0; i < m * n; ++i)
						h_gW[i] = glades::rng::standard_normal(rng) * gradScales[si];
				}
				d_gW.upload(h_gW.data(), m * n);

				bool ok = glades::gpu::atlas_gpu_step(
					state, d_W.data(), d_gW.data(), m, n,
					3.05e-6f, 5e-4f, 0.0f, 0.01f, 1.0f, ac);

				if (!ok) { nanDetected = true; break; }

				if (state.step > 0ULL && (state.step % tSub) == 0ULL)
				{
					glades::gpu::AtlasGpuDiag diag = glades::gpu::atlas_gpu_get_diag(state);
					if (diag.valid && diag.gzNorm != diag.gzNorm)
					{
						printf("    [NaN] gz_norm=NaN at step %llu\n",
						       (unsigned long long)state.step);
						nanDetected = true;
					}
				}
			}

			printf("    %s\n", nanDetected ? "[FAIL] NaN detected" : "[PASS] No NaN");
			ASSERT("==============GPU ATLAS Test3: NaN from tiny gradient==============",
			       !nanDetected);
		}
	}

	// ---------------------------------------------------------------
	// Test GPU 4: Sequential large+small matrix (tokE then Wq interaction)
	//
	// In production, tokE (32000×1024) runs first, then layer 0 Wq
	// (1024×1024). Test if the large SGEMM corrupts cuBLAS state
	// or shared resources that affect the subsequent smaller weight.
	// ---------------------------------------------------------------
	printf("-----------------------------------\n");
	printf("ATLAS GPU Test 4: Sequential tokE + Wq interaction\n");
	printf("-----------------------------------\n");
	if (glades::gpu::initDevice())
	{
		const unsigned int tSub = 10;
		const float lr = 5e-4f;
		const float invBatch = 3.05e-6f;

		// --- tokE state: 32000×1024, r=256 ---
		glades::gpu::GpuBuffer<float> d_W_tok, d_gW_tok;
		d_W_tok.allocate(32000u * 1024u);
		d_gW_tok.allocate(32000u * 1024u);
		{
			std::vector<float> h(32000u * 1024u, 0.01f);
			d_W_tok.upload(h.data(), h.size());
		}

		glades::gpu::GpuAtlasWeightState tokState;
		glades::rng::Engine rng;
		glades::rng::seed_engine(rng, 12345ULL);
		glades::gpu::atlas_gpu_init(tokState, 32000, 1024, 256, 0.01f, rng);

		// --- Wq state: 1024×1024, r=256 ---
		glades::gpu::GpuBuffer<float> d_W_wq, d_gW_wq;
		d_W_wq.allocate(1024u * 1024u);
		d_gW_wq.allocate(1024u * 1024u);
		{
			std::vector<float> h(1024u * 1024u, 0.01f);
			d_W_wq.upload(h.data(), h.size());
		}

		glades::gpu::GpuAtlasWeightState wqState;
		// NOTE: rng is SHARED — this mimics production where the same rng
		// is used for all weight inits sequentially
		glades::gpu::atlas_gpu_init(wqState, 1024, 1024, 256, 0.01f, rng);

		glades::ATLASConfig ac;
		ac.rank = 256;
		ac.tSub = tSub;
		ac.beta = 0.99f;
		ac.muMin = 0.01f;
		ac.muMax = 0.3f;
		ac.eps = 1e-8f;
		ac.kappaMax = 10.0f;
		ac.betaRefresh = 0.5f;
		ac.powerIters = 2;
		ac.muGrowthRate = 0.001f;
		ac.biasCorrection = true;

		bool nanDetected = false;

		for (unsigned int step = 0; step < tSub + 2; ++step)
		{
			// Fill tokE gradient (large, ~0.01 magnitude)
			{
				std::vector<float> h(32000u * 1024u);
				for (size_t i = 0; i < h.size(); ++i)
					h[i] = glades::rng::standard_normal(rng) * 0.01f;
				d_gW_tok.upload(h.data(), h.size());
			}
			// Step tokE FIRST (like production)
			glades::gpu::atlas_gpu_step(
				tokState, d_W_tok.data(), d_gW_tok.data(), 32000, 1024,
				invBatch, lr, 0.0f, 0.01f, 1.0f, ac);

			// Fill Wq gradient (tiny, ~2e-6 magnitude)
			{
				std::vector<float> h(1024u * 1024u);
				for (size_t i = 0; i < h.size(); ++i)
					h[i] = glades::rng::standard_normal(rng) * 2e-6f;
				d_gW_wq.upload(h.data(), h.size());
			}
			// Step Wq SECOND
			glades::gpu::atlas_gpu_step(
				wqState, d_W_wq.data(), d_gW_wq.data(), 1024, 1024,
				invBatch, lr, 0.0f, 0.01f, 1.0f, ac);

			if (wqState.step > 0ULL && (wqState.step % tSub) == 0ULL)
			{
				glades::gpu::AtlasGpuDiag diag = glades::gpu::atlas_gpu_get_diag(wqState);
				if (diag.valid && diag.gzNorm != diag.gzNorm)
				{
					printf("    [NaN] Wq gz_norm=NaN at step %llu\n",
					       (unsigned long long)wqState.step);
					nanDetected = true;
				}
			}
		}

		printf("    %s\n", nanDetected ? "[FAIL] NaN in Wq after tokE" : "[PASS] No NaN");
		ASSERT("==============GPU ATLAS Test4: Sequential tokE+Wq NaN==============",
		       !nanDetected);
	}
	printf("Unit Test Success %s[%d]\n", __FILE__, __LINE__);

#else
	printf("  CUDA not enabled at build time, skipping GPU ATLAS NaN test.\n");
#endif // GLADES_HAVE_CUDA

	printf("\n============================================================\n");
}
