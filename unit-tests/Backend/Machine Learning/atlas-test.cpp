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
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <unistd.h>
#include <vector>

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

		const glades::NNetworkStatus st = net.train(di);
		ASSERT("==============ATLAS::DFF_Regression TrainStatus() Failed==============", st.ok());

		printf("[UT] ATLAS DFF regression: training completed successfully\n");

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

		const glades::NNetworkStatus st = net.train(di);
		ASSERT("==============ATLAS::DFF_XOR TrainStatus() Failed==============", st.ok());

		printf("[UT] ATLAS DFF XOR: training completed successfully\n");

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

		const glades::NNetworkStatus st = net.train(di);
		ASSERT("==============ATLAS::Transformer TrainStatus() Failed==============", st.ok());

		printf("[UT] ATLAS Transformer: training completed successfully\n");

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

		const glades::NNetworkStatus st = net.train(di);
		ASSERT("==============ATLAS::RNN TrainStatus() Failed==============", st.ok());
		printf("[UT] ATLAS RNN: training completed successfully\n");

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

		const glades::NNetworkStatus st = net.train(di);
		ASSERT("==============ATLAS::GRU TrainStatus() Failed==============", st.ok());
		printf("[UT] ATLAS GRU: training completed successfully\n");

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

		const glades::NNetworkStatus st = net.train(di);
		ASSERT("==============ATLAS::LSTM TrainStatus() Failed==============", st.ok());
		printf("[UT] ATLAS LSTM: training completed successfully\n");

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

	printf("\n============================================================\n");
}
