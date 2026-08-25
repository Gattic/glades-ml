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
#include "nn-mixed-precision-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"

#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>

namespace {

static bool read_file_to_string(const std::string& path, std::string& out)
{
	out.clear();
	std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
	if (!in)
		return false;
	std::string s;
	char buf[4096];
	while (in.good())
	{
		in.read(buf, sizeof(buf));
		const std::streamsize n = in.gcount();
		if (n > 0)
			s.append(buf, static_cast<size_t>(n));
	}
	out.swap(s);
	return true;
}

static bool mkdir_if_missing(const std::string& path)
{
	if (path.empty())
		return false;
	if (::mkdir(path.c_str(), 0777) == 0)
		return true;
	return true; // EEXIST ok
}

static bool parse_kv_manifest(const std::string& path, std::map<std::string, std::string>& outKv)
{
	outKv.clear();
	std::ifstream in(path.c_str(), std::ios::in | std::ios::binary);
	if (!in)
		return false;
	std::string line;
	bool firstLine = true;
	while (std::getline(in, line))
	{
		if (firstLine)
		{
			outKv["__magic__"] = line;
			firstLine = false;
			continue;
		}
		if (line.empty())
			continue;
		const size_t eq = line.find('=');
		if (eq == std::string::npos)
			continue;
		outKv[line.substr(0, eq)] = line.substr(eq + 1);
	}
	return true;
}

static bool kv_get_f32(const std::map<std::string, std::string>& kv, const std::string& key, float& out)
{
	std::map<std::string, std::string>::const_iterator it = kv.find(key);
	if (it == kv.end())
		return false;
	out = static_cast<float>(atof(it->second.c_str()));
	return std::isfinite(out) != 0;
}

struct NoopCallbacks : public glades::ITrainingCallbacks
{
	virtual void onRunStart(const glades::NNetwork&, int) {}
	virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics&) { return false; }
	virtual void onRunEnd(const glades::NNetwork&, int) {}
};

static glades::NNInfo* make_tiny_transformer_regression_info(const std::string& name,
                                                            unsigned int dModel,
                                                            unsigned int outSize,
                                                            float learningRate)
{
	auto in = shmea::make_gpointer<glades::InputLayerInfo>(
	    /*batchSize*/ 1,
	    /*learningRate*/ learningRate,
	    /*momentumFactor*/ 0.0f,
	    /*weightDecay1*/ 0.0f,
	    /*weightDecay2*/ 0.0f,
	    /*pDropout*/ 0.0f,
	    /*activationType*/ glades::GMath::LINEAR,
	    /*activationParam*/ 1.0f);

	std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
	hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
	    /*size*/ static_cast<int>(dModel),
	    /*learningRate*/ learningRate,
	    /*momentumFactor*/ 0.0f,
	    /*weightDecay1*/ 0.0f,
	    /*weightDecay2*/ 0.0f,
	    /*pDropout*/ 0.0f,
	    /*activationType*/ glades::GMath::LINEAR,
	    /*activationParam*/ 1.0f));

	auto out = shmea::make_gpointer<glades::OutputLayerInfo>(static_cast<int>(outSize), glades::OutputLayerInfo::REGRESSION);
	return new glades::NNInfo(name.c_str(), in, hidden, out);
}

static glades::NumberInput* make_single_sequence_dataset(unsigned int T,
                                                         unsigned int inSize,
                                                         unsigned int outSize,
                                                         float inputScale,
                                                         float expectedScale)
{
	glades::NumberInput* di = new glades::NumberInput();
	di->trainMatrix = shmea::GMatrix(static_cast<int>(T), shmea::GVector<float>(static_cast<int>(inSize), 0.0f));
	di->trainExpectedMatrix = shmea::GMatrix(static_cast<int>(T), shmea::GVector<float>(static_cast<int>(outSize), 0.0f));
	for (unsigned int t = 0; t < T; ++t)
	{
		for (unsigned int i = 0; i < inSize; ++i)
			di->trainMatrix[static_cast<int>(t)][static_cast<int>(i)] = inputScale * static_cast<float>(t + 1u);
		for (unsigned int k = 0; k < outSize; ++k)
			di->trainExpectedMatrix[static_cast<int>(t)][static_cast<int>(k)] = expectedScale * static_cast<float>(t + 1u);
	}
	di->testMatrix = di->trainMatrix;
	di->testExpectedMatrix = di->trainExpectedMatrix;
	return di;
}

static std::string model_weights_path(const std::string& modelName)
{
	return "database/models/" + modelName + "/weights.bin";
}

} // namespace

void NNMixedPrecisionUnitTest()
{
	printf("============================================================\n");
	printf("-----------------------------------\n");
	printf("NN Mixed Precision Test (Transformer)\n");
	printf("-----------------------------------\n");

	NoopCallbacks cb;

	// Ensure database directories exist (tests share this location).
	mkdir_if_missing("database");
	mkdir_if_missing("database/models");
	mkdir_if_missing("database/checkpoints");

	// ============================
	// Case A: saveModel manifest persists mixed-precision TrainingConfig
	// ============================
	{
		glades::NumberInput* di = make_single_sequence_dataset(/*T*/ 4u, /*in*/ 1u, /*out*/ 1u, /*x*/ 0.1f, /*y*/ 0.2f);
		glades::NNInfo* info = make_tiny_transformer_regression_info("ut_mp_manifest", /*dModel*/ 8u, /*out*/ 1u, /*lr*/ 0.0f);
		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
		net.setSeed(123u);

		glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
		cfg.transformer.nHeadsOverride = 2;
		cfg.transformer.dFFOverride = 32;
		cfg.optimizer.type = glades::OptimizerConfig::ADAMW;

		cfg.mixedPrecision.enable = true;
		cfg.mixedPrecision.weightDType = glades::MixedPrecisionConfig::WEIGHT_BF16;
		cfg.mixedPrecision.useLossScaling = true;
		cfg.mixedPrecision.dynamicLossScaling = true;
		cfg.mixedPrecision.lossScaleInit = 128.0f;
		cfg.mixedPrecision.lossScaleMin = 1.0f;
		cfg.mixedPrecision.lossScaleMax = 1024.0f;
		cfg.mixedPrecision.growthInterval = 7;
		cfg.mixedPrecision.growthFactor = 2.0f;
		cfg.mixedPrecision.backoffFactor = 0.5f;

		ASSERT("==============NNMixedPrecision::A_InitTestStatus() Failed==============", net.test(di).ok());

		const std::string modelName = "ut_model_pkg_mp_manifest";
		ASSERT("==============NNMixedPrecision::A_SaveModel() Failed==============", net.saveModel(modelName).ok());

		std::map<std::string, std::string> kv;
		ASSERT("==============NNMixedPrecision::A_ParseManifest() Failed==============",
		       parse_kv_manifest("database/models/" + modelName + "/manifest.txt", kv));
		ASSERT("==============NNMixedPrecision::A_Magic() Failed==============", kv["__magic__"] == "GLADES_MODEL");
		ASSERT("==============NNMixedPrecision::A_HasEnable() Failed==============", kv["training.mixedPrecision.enable"] == "1");
		ASSERT("==============NNMixedPrecision::A_HasDType() Failed==============", kv["training.mixedPrecision.weightDType"] == "2"); // BF16
		ASSERT("==============NNMixedPrecision::A_HasLossScaling() Failed==============", kv["training.mixedPrecision.useLossScaling"] == "1");
		ASSERT("==============NNMixedPrecision::A_HasDynamic() Failed==============", kv["training.mixedPrecision.dynamicLossScaling"] == "1");

		// Round-trip through loadModel restores TrainingConfig fields.
		glades::NNetwork net2(glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
		ASSERT("==============NNMixedPrecision::A_LoadModel() Failed==============", net2.loadModel(modelName, di).ok());
		const glades::TrainingConfig& cfg2 = net2.getTrainingConfig();
		ASSERT("==============NNMixedPrecision::A_LoadRestoresMPEnable() Failed==============", cfg2.mixedPrecision.enable == true);
		ASSERT("==============NNMixedPrecision::A_LoadRestoresMPDType() Failed==============",
		       cfg2.mixedPrecision.weightDType == glades::MixedPrecisionConfig::WEIGHT_BF16);
		ASSERT("==============NNMixedPrecision::A_LoadRestoresLossScaleInit() Failed==============",
		       fabs(cfg2.mixedPrecision.lossScaleInit - 128.0f) < 1e-6f);
		ASSERT("==============NNMixedPrecision::A_LoadRestoresGrowthInterval() Failed==============", cfg2.mixedPrecision.growthInterval == 7);

		delete di;
		delete info;
	}

	// ============================
	// Case B: Mixed-precision training produces different weights than FP32 training
	// ============================
	{
		glades::NumberInput* di = make_single_sequence_dataset(/*T*/ 6u, /*in*/ 1u, /*out*/ 1u, /*x*/ 0.05f, /*y*/ 0.1f);
		glades::NNInfo* infoA = make_tiny_transformer_regression_info("ut_mp_train_f32", /*dModel*/ 8u, /*out*/ 1u, /*lr*/ 0.01f);
		glades::NNInfo* infoB = make_tiny_transformer_regression_info("ut_mp_train_mp", /*dModel*/ 8u, /*out*/ 1u, /*lr*/ 0.01f);

		// FP32 baseline
		glades::NNetwork netF32(infoA, glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
		netF32.setSeed(777u);
		netF32.getTerminatorMutable().setEpoch(1);
		netF32.getTerminatorMutable().setAccuracy(0);
		{
			glades::TrainingConfig& cfg = netF32.getTrainingConfigMutable();
			cfg.transformer.nHeadsOverride = 2;
			cfg.transformer.dFFOverride = 32;
			cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
			cfg.mixedPrecision.enable = false;
		}
		ASSERT("==============NNMixedPrecision::B_InitF32() Failed==============", netF32.test(di).ok());
		ASSERT("==============NNMixedPrecision::B_TrainF32() Failed==============", netF32.train(di, &cb).ok());
		const std::string modelF32 = "ut_model_pkg_mp_train_f32";
		ASSERT("==============NNMixedPrecision::B_SaveF32() Failed==============", netF32.saveModel(modelF32).ok());

		// Mixed precision (FP16)
		glades::NNetwork netMP(infoB, glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
		netMP.setSeed(777u); // same seed so only dtype path causes divergence
		netMP.getTerminatorMutable().setEpoch(1);
		netMP.getTerminatorMutable().setAccuracy(0);
		{
			glades::TrainingConfig& cfg = netMP.getTrainingConfigMutable();
			cfg.transformer.nHeadsOverride = 2;
			cfg.transformer.dFFOverride = 32;
			cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
			cfg.mixedPrecision.enable = true;
			cfg.mixedPrecision.weightDType = glades::MixedPrecisionConfig::WEIGHT_F16;
			cfg.mixedPrecision.useLossScaling = true;
			cfg.mixedPrecision.dynamicLossScaling = false; // fixed scaling for determinism
			cfg.mixedPrecision.lossScaleInit = 1024.0f;
		}
		ASSERT("==============NNMixedPrecision::B_InitMP() Failed==============", netMP.test(di).ok());
		ASSERT("==============NNMixedPrecision::B_TrainMP() Failed==============", netMP.train(di, &cb).ok());
		const std::string modelMP = "ut_model_pkg_mp_train_f16";
		ASSERT("==============NNMixedPrecision::B_SaveMP() Failed==============", netMP.saveModel(modelMP).ok());

		std::string wF32, wMP;
		ASSERT("==============NNMixedPrecision::B_ReadWeightsF32() Failed==============", read_file_to_string(model_weights_path(modelF32), wF32));
		ASSERT("==============NNMixedPrecision::B_ReadWeightsMP() Failed==============", read_file_to_string(model_weights_path(modelMP), wMP));
		ASSERT("==============NNMixedPrecision::B_WeightsDifferF32vsMP() Failed==============", wF32 != wMP);

		delete di;
		delete infoA;
		delete infoB;
	}

	// ============================
	// Case C: Checkpoint persists loss-scale state (growth) and restores it on load
	// ============================
	{
		glades::NumberInput* di = make_single_sequence_dataset(/*T*/ 4u, /*in*/ 1u, /*out*/ 1u, /*x*/ 0.2f, /*y*/ 0.1f);
		glades::NNInfo* info = make_tiny_transformer_regression_info("ut_mp_ckpt_growth", /*dModel*/ 8u, /*out*/ 1u, /*lr*/ 0.01f);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
		net.setSeed(2026u);
		net.getTerminatorMutable().setEpoch(1);
		net.getTerminatorMutable().setAccuracy(0);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.transformer.nHeadsOverride = 2;
			cfg.transformer.dFFOverride = 32;
			cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
			cfg.perElementGradClip = 0.0f; // no clipping to avoid interacting with scale
			cfg.mixedPrecision.enable = true;
			cfg.mixedPrecision.weightDType = glades::MixedPrecisionConfig::WEIGHT_BF16;
			cfg.mixedPrecision.useLossScaling = true;
			cfg.mixedPrecision.dynamicLossScaling = true;
			cfg.mixedPrecision.lossScaleInit = 128.0f;
			cfg.mixedPrecision.lossScaleMin = 1.0f;
			cfg.mixedPrecision.lossScaleMax = 1024.0f;
			cfg.mixedPrecision.growthInterval = 1; // grow every successful step
			cfg.mixedPrecision.growthFactor = 2.0f;
			cfg.mixedPrecision.backoffFactor = 0.5f;
		}

		ASSERT("==============NNMixedPrecision::C_Init() Failed==============", net.test(di).ok());
		ASSERT("==============NNMixedPrecision::C_Train() Failed==============", net.train(di, &cb).ok());

		glades::NNetwork::CheckpointConfig ccfg;
		ccfg.maxShardBytes = 4096u;
		ccfg.includeOptimizerState = true;

		const std::string ckptName = "ut_checkpoint_mp_growth";
		ASSERT("==============NNMixedPrecision::C_SaveCheckpoint() Failed==============", net.saveCheckpoint(ckptName, ccfg).ok());

		std::map<std::string, std::string> kv;
		ASSERT("==============NNMixedPrecision::C_ParseCkptManifest() Failed==============",
		       parse_kv_manifest("database/checkpoints/" + ckptName + "/manifest.txt", kv));
		ASSERT("==============NNMixedPrecision::C_CkptMagic() Failed==============", kv["__magic__"] == "GLADES_CHECKPOINT");

		float lossScale = 0.0f;
		ASSERT("==============NNMixedPrecision::C_ReadLossScale() Failed==============", kv_get_f32(kv, "transformer.lossScale", lossScale));
		// growthInterval=1 => 128 -> 256 after first good step (clamped by max=1024)
		ASSERT("==============NNMixedPrecision::C_LossScaleGrew() Failed==============", fabs(lossScale - 256.0f) < 1e-3f);

		// Load checkpoint and re-save; lossScale should round-trip.
		glades::NNetwork net2(glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
		ASSERT("==============NNMixedPrecision::C_LoadCheckpoint() Failed==============", net2.loadCheckpoint(ckptName, di).ok());
		const std::string ckptName2 = "ut_checkpoint_mp_growth_roundtrip";
		ASSERT("==============NNMixedPrecision::C_SaveCheckpoint2() Failed==============", net2.saveCheckpoint(ckptName2, ccfg).ok());
		std::map<std::string, std::string> kv2;
		ASSERT("==============NNMixedPrecision::C_ParseCkptManifest2() Failed==============",
		       parse_kv_manifest("database/checkpoints/" + ckptName2 + "/manifest.txt", kv2));
		float lossScale2 = 0.0f;
		ASSERT("==============NNMixedPrecision::C_ReadLossScale2() Failed==============", kv_get_f32(kv2, "transformer.lossScale", lossScale2));
		ASSERT("==============NNMixedPrecision::C_LossScaleRoundTrip() Failed==============", fabs(lossScale2 - 256.0f) < 1e-3f);

		// Also verify TrainingConfig mixed-precision keys persist in checkpoint manifest.
		ASSERT("==============NNMixedPrecision::C_CkptHasMPEnable() Failed==============", kv["training.mixedPrecision.enable"] == "1");
		ASSERT("==============NNMixedPrecision::C_CkptHasMPDType() Failed==============", kv["training.mixedPrecision.weightDType"] == "2"); // BF16

		delete di;
		delete info;
	}

	// ============================
	// Case D: Dynamic loss scaling backoff on overflowed gradients (skip-step)
	// ============================
	{
		// Construct a sequence with huge expected values and disable per-element clipping so
		// dLogits can overflow when multiplied by lossScaleInit.
		//
		// IMPORTANT:
		// Keep the *unscaled* loss finite so the init/test pass succeeds (MSE uses squared error).
		// Then pick a lossScaleInit large enough that (dLogits * lossScaleInit) overflows to Inf.
		glades::NumberInput* di = make_single_sequence_dataset(/*T*/ 2u, /*in*/ 1u, /*out*/ 1u, /*x*/ 1.0f, /*y*/ 1e10f);
		glades::NNInfo* info = make_tiny_transformer_regression_info("ut_mp_ckpt_backoff", /*dModel*/ 8u, /*out*/ 1u, /*lr*/ 0.01f);

		glades::NNetwork net(info, glades::NNetwork::TYPE_TRANSFORMER_ENCODER);
		net.setSeed(31337u);
		net.getTerminatorMutable().setEpoch(1);
		net.getTerminatorMutable().setAccuracy(0);
		{
			glades::TrainingConfig& cfg = net.getTrainingConfigMutable();
			cfg.transformer.nHeadsOverride = 2;
			cfg.transformer.dFFOverride = 32;
			cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
			cfg.perElementGradClip = 0.0f; // disable clipf_maybe
			cfg.mixedPrecision.enable = true;
			cfg.mixedPrecision.weightDType = glades::MixedPrecisionConfig::WEIGHT_F16;
			cfg.mixedPrecision.useLossScaling = true;
			cfg.mixedPrecision.dynamicLossScaling = true;
			cfg.mixedPrecision.lossScaleInit = 1e30f;
			cfg.mixedPrecision.lossScaleMin = 1.0f;
			cfg.mixedPrecision.lossScaleMax = 1e30f;
			cfg.mixedPrecision.growthInterval = 1000000;
			cfg.mixedPrecision.growthFactor = 2.0f;
			cfg.mixedPrecision.backoffFactor = 0.5f;
		}

		ASSERT("==============NNMixedPrecision::D_Init() Failed==============", net.test(di).ok());
		ASSERT("==============NNMixedPrecision::D_Train() Failed==============", net.train(di, &cb).ok());

		glades::NNetwork::CheckpointConfig ccfg;
		ccfg.maxShardBytes = 4096u;
		ccfg.includeOptimizerState = true;
		const std::string ckptName = "ut_checkpoint_mp_backoff";
		ASSERT("==============NNMixedPrecision::D_SaveCheckpoint() Failed==============", net.saveCheckpoint(ckptName, ccfg).ok());

		std::map<std::string, std::string> kv;
		ASSERT("==============NNMixedPrecision::D_ParseManifest() Failed==============",
		       parse_kv_manifest("database/checkpoints/" + ckptName + "/manifest.txt", kv));
		float lossScale = 0.0f;
		ASSERT("==============NNMixedPrecision::D_ReadLossScale() Failed==============", kv_get_f32(kv, "transformer.lossScale", lossScale));
		ASSERT("==============NNMixedPrecision::D_LossScaleBackedOff() Failed==============", lossScale < 1e30f);

		delete di;
		delete info;
	}

	printf("\n============================================================\n");
}

