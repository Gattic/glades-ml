#include "hyperparameter-tuner-test.h"
#include "../../unit-test.h"
#include "../../../Backend/Machine Learning/Networks/hyperparameter_tuner.h"
#include "../../../Backend/Machine Learning/Networks/training_config.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include <cmath>
#include <vector>
#include <cstdio>
#include <limits>

void SearchSpaceUnitTest()
{
	printf("=== SearchSpace Unit Tests ===\n");

	// Build a 4-param space:
	// 0: log-continuous [1e-4, 1e-1]
	// 1: linear-continuous [0.0, 10.0]
	// 2: integer [1, 100]
	// 3: categorical {0.0, 1.0, 2.0}
	glades::SearchSpace space;

	{
		glades::HyperParameter hp;
		hp.name = "logParam";
		hp.type = glades::HyperParameter::CONTINUOUS;
		hp.low = 1e-4f;
		hp.high = 1e-1f;
		hp.logScale = true;
		space.params.push_back(hp);
	}
	{
		glades::HyperParameter hp;
		hp.name = "linearParam";
		hp.type = glades::HyperParameter::CONTINUOUS;
		hp.low = 0.0f;
		hp.high = 10.0f;
		hp.logScale = false;
		space.params.push_back(hp);
	}
	{
		glades::HyperParameter hp;
		hp.name = "intParam";
		hp.type = glades::HyperParameter::INTEGER;
		hp.low = 1.0f;
		hp.high = 100.0f;
		hp.logScale = false;
		space.params.push_back(hp);
	}
	{
		glades::HyperParameter hp;
		hp.name = "catParam";
		hp.type = glades::HyperParameter::CATEGORICAL;
		hp.low = 0.0f;
		hp.high = 2.0f;
		hp.logScale = false;
		hp.choices.push_back(0.0f);
		hp.choices.push_back(1.0f);
		hp.choices.push_back(2.0f);
		space.params.push_back(hp);
	}

	G_assert(__FILE__, __LINE__,
	         "==============SearchSpace::dimensions==============",
	         space.dimensions() == 4);

	// ---- Sub-test 1: encode -> decode round-trip ----
	printf("  Sub-test 1: encode -> decode round-trip\n");
	{
		std::vector<float> raw(4);
		raw[0] = 1e-3f;   // log-continuous mid value
		raw[1] = 5.0f;    // linear-continuous mid
		raw[2] = 50.0f;   // integer mid
		raw[3] = 1.0f;    // categorical choice index 1

		std::vector<float> normalized = space.encode(raw);
		std::vector<float> decoded = space.decode(normalized);

		printf("    logParam:    raw=%e -> norm=%f -> decoded=%e\n", raw[0], normalized[0], decoded[0]);
		printf("    linearParam: raw=%f -> norm=%f -> decoded=%f\n", raw[1], normalized[1], decoded[1]);
		printf("    intParam:    raw=%f -> norm=%f -> decoded=%f\n", raw[2], normalized[2], decoded[2]);
		printf("    catParam:    raw=%f -> norm=%f -> decoded=%f\n", raw[3], normalized[3], decoded[3]);

		// Log-continuous: allow 5% relative error due to float precision
		float logRelErr = fabsf(decoded[0] - raw[0]) / raw[0];
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::round-trip logParam==============",
		         logRelErr < 0.05f);

		// Linear-continuous: very tight
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::round-trip linearParam==============",
		         fabsf(decoded[1] - raw[1]) < 0.01f);

		// Integer: should round-trip exactly
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::round-trip intParam==============",
		         fabsf(decoded[2] - raw[2]) < 0.5f);

		// Categorical: exact match
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::round-trip catParam==============",
		         fabsf(decoded[3] - raw[3]) < 0.01f);
	}

	// ---- Sub-test 2: boundary values ----
	printf("  Sub-test 2: boundary values (decode(0)=low, decode(1)=high)\n");
	{
		// decode(0,...) should give low values
		std::vector<float> zeros(4, 0.0f);
		std::vector<float> decodedLow = space.decode(zeros);

		printf("    decode(0): logParam=%e (expect %e)\n", decodedLow[0], 1e-4f);
		printf("    decode(0): linearParam=%f (expect %f)\n", decodedLow[1], 0.0f);
		printf("    decode(0): intParam=%f (expect %f)\n", decodedLow[2], 1.0f);
		printf("    decode(0): catParam=%f (expect %f)\n", decodedLow[3], 0.0f);

		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::decode(0) logParam==low==============",
		         fabsf(decodedLow[0] - 1e-4f) < 1e-5f);
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::decode(0) linearParam==low==============",
		         fabsf(decodedLow[1] - 0.0f) < 0.01f);
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::decode(0) intParam==low==============",
		         fabsf(decodedLow[2] - 1.0f) < 0.5f);
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::decode(0) catParam==first==============",
		         fabsf(decodedLow[3] - 0.0f) < 0.01f);

		// decode(1,...) should give high values
		std::vector<float> ones(4, 1.0f);
		std::vector<float> decodedHigh = space.decode(ones);

		printf("    decode(1): logParam=%e (expect %e)\n", decodedHigh[0], 1e-1f);
		printf("    decode(1): linearParam=%f (expect %f)\n", decodedHigh[1], 10.0f);
		printf("    decode(1): intParam=%f (expect %f)\n", decodedHigh[2], 100.0f);
		printf("    decode(1): catParam=%f (expect %f)\n", decodedHigh[3], 2.0f);

		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::decode(1) logParam==high==============",
		         fabsf(decodedHigh[0] - 1e-1f) < 1e-3f);
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::decode(1) linearParam==high==============",
		         fabsf(decodedHigh[1] - 10.0f) < 0.01f);
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::decode(1) intParam==high==============",
		         fabsf(decodedHigh[2] - 100.0f) < 0.5f);
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::decode(1) catParam==last==============",
		         fabsf(decodedHigh[3] - 2.0f) < 0.01f);
	}

	// ---- Sub-test 3: defaultTrainingSearchSpace ----
	printf("  Sub-test 3: defaultTrainingSearchSpace\n");
	{
		glades::SearchSpace defSpace = glades::SearchSpace::defaultTrainingSearchSpace();
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::defaultTrainingSearchSpace has 9 params==============",
		         defSpace.dimensions() == 9);
		printf("    Default search space has %u dimensions\n", defSpace.dimensions());
	}

	// ---- Sub-test 4: applyToConfig ----
	printf("  Sub-test 4: applyToConfig\n");
	{
		glades::SearchSpace defSpace = glades::SearchSpace::defaultTrainingSearchSpace();
		glades::TrainingConfig baseCfg;

		// Create raw values matching the default space order:
		// 0: learningRate (skipped), 1: optimizer, 2: lrScheduleType,
		// 3: lrScheduleGamma, 4: adamBeta1, 5: adamBeta2,
		// 6: globalGradClipNorm, 7: warmupSteps, 8: weightDecay (skipped)
		std::vector<float> raw(9);
		raw[0] = 0.001f;   // learningRate (skipped)
		raw[1] = 1.0f;     // optimizer = ADAMW
		raw[2] = 3.0f;     // lrScheduleType = COSINE
		raw[3] = 0.5f;     // lrScheduleGamma
		raw[4] = 0.9f;     // adamBeta1
		raw[5] = 0.999f;   // adamBeta2
		raw[6] = 5.0f;     // globalGradClipNorm
		raw[7] = 1000.0f;  // warmupSteps
		raw[8] = 0.01f;    // weightDecay (skipped)

		glades::TrainingConfig result = defSpace.applyToConfig(raw, baseCfg);

		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::applyToConfig optimizer==ADAMW==============",
		         result.optimizer.type == glades::OptimizerConfig::ADAMW);
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::applyToConfig lrSchedule==COSINE==============",
		         result.lrSchedule.type == glades::LearningRateScheduleConfig::COSINE);
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::applyToConfig gamma==============",
		         fabsf(result.lrSchedule.gamma - 0.5f) < 0.01f);
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::applyToConfig adamBeta1==============",
		         fabsf(result.optimizer.adamBeta1 - 0.9f) < 0.01f);
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::applyToConfig globalGradClipNorm==============",
		         fabsf(result.globalGradClipNorm - 5.0f) < 0.01f);
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::applyToConfig warmupSteps==============",
		         result.warmup.warmupSteps == 1000);
		G_assert(__FILE__, __LINE__,
		         "==============SearchSpace::applyToConfig warmupType==LINEAR==============",
		         result.warmup.type == glades::WarmupConfig::WARMUP_LINEAR);
	}

	printf("=== All SearchSpace Unit Tests PASSED ===\n");
}

void HyperparameterTunerUnitTest()
{
	printf("=== HyperparameterTuner Unit Tests ===\n");

	// Build a simple 1-param search space for testing
	glades::SearchSpace space;
	{
		glades::HyperParameter hp;
		hp.name = "testParam";
		hp.type = glades::HyperParameter::CONTINUOUS;
		hp.low = 0.0f;
		hp.high = 1.0f;
		hp.logScale = false;
		space.params.push_back(hp);
	}

	glades::HyperparameterTuner tuner(space, 5, 10);

	// ---- Sub-test 1: getMaxTrials ----
	printf("  Sub-test 1: getMaxTrials\n");
	G_assert(__FILE__, __LINE__,
	         "==============HPTuner::getMaxTrials==5==============",
	         tuner.getMaxTrials() == 5);

	// ---- Sub-test 2: getEpochsPerTrial ----
	printf("  Sub-test 2: getEpochsPerTrial\n");
	G_assert(__FILE__, __LINE__,
	         "==============HPTuner::getEpochsPerTrial==10==============",
	         tuner.getEpochsPerTrial() == 10);

	// ---- Sub-test 3: suggestNext returns 1-dim vector in [0,1] ----
	printf("  Sub-test 3: suggestNext returns valid point\n");
	{
		std::vector<float> suggestion = tuner.suggestNext();
		G_assert(__FILE__, __LINE__,
		         "==============HPTuner::suggestNext dim==1==============",
		         suggestion.size() == 1);
		G_assert(__FILE__, __LINE__,
		         "==============HPTuner::suggestNext in [0,1]==============",
		         suggestion[0] >= 0.0f && suggestion[0] <= 1.0f);
		printf("    suggestion[0] = %f\n", suggestion[0]);
	}

	// ---- Sub-test 4: reportResult tracks best score ----
	printf("  Sub-test 4: reportResult tracks best score\n");
	{
		// Report a few results with decreasing loss
		std::vector<float> p1(1, 0.3f);
		tuner.reportResult(p1, 2.5f);

		std::vector<float> p2(1, 0.7f);
		tuner.reportResult(p2, 1.0f);

		std::vector<float> p3(1, 0.5f);
		tuner.reportResult(p3, 3.0f);

		// Best score should be the minimum (1.0f)
		printf("    bestScore = %f (expect 1.0)\n", tuner.getBestScore());
		G_assert(__FILE__, __LINE__,
		         "==============HPTuner::bestScore==1.0==============",
		         fabsf(tuner.getBestScore() - 1.0f) < 0.01f);

		// Trials should have 3 entries
		G_assert(__FILE__, __LINE__,
		         "==============HPTuner::trials size==3==============",
		         tuner.getTrials().size() == 3);

		// All trials should be completed
		for (unsigned int i = 0; i < tuner.getTrials().size(); ++i)
		{
			G_assert(__FILE__, __LINE__,
			         "==============HPTuner::trial completed==============",
			         tuner.getTrials()[i].completed);
		}
	}

	// ---- Sub-test 5: suggestNext uses BO after initial random phase ----
	printf("  Sub-test 5: suggestNext after initial random phase\n");
	{
		// We've reported 3 results. nInitialRandom_ defaults to 5 but capped to maxTrials(5).
		// Report 2 more to get past nInitialRandom_
		std::vector<float> p4(1, 0.2f);
		tuner.reportResult(p4, 1.5f);

		std::vector<float> p5(1, 0.8f);
		tuner.reportResult(p5, 0.5f);

		// Now trialCounter_ == 5 which >= nInitialRandom_(5), so next should use BO
		std::vector<float> boSuggestion = tuner.suggestNext();
		G_assert(__FILE__, __LINE__,
		         "==============HPTuner::BO suggestion dim==1==============",
		         boSuggestion.size() == 1);
		G_assert(__FILE__, __LINE__,
		         "==============HPTuner::BO suggestion in [0,1]==============",
		         boSuggestion[0] >= 0.0f && boSuggestion[0] <= 1.0f);
		printf("    BO suggestion[0] = %f\n", boSuggestion[0]);

		// Best score should now be 0.5
		printf("    bestScore = %f (expect 0.5)\n", tuner.getBestScore());
		G_assert(__FILE__, __LINE__,
		         "==============HPTuner::bestScore==0.5==============",
		         fabsf(tuner.getBestScore() - 0.5f) < 0.01f);
	}

	printf("=== All HyperparameterTuner Unit Tests PASSED ===\n");
}

void BayesianLRScheduleTest()
{
	printf("=== BayesianLRSchedule Unit Tests ===\n");

	// ---- Sub-test 1: BAYESIAN enum exists and multiplier(0) returns 1.0 ----
	printf("  Sub-test 1: BAYESIAN enum and multiplier\n");
	{
		glades::LearningRateScheduleConfig cfg;
		cfg.type = glades::LearningRateScheduleConfig::BAYESIAN;

		G_assert(__FILE__, __LINE__,
		         "==============BayesianLR::enum value==4==============",
		         static_cast<int>(glades::LearningRateScheduleConfig::BAYESIAN) == 4);

		float m = cfg.multiplier(0);
		G_assert(__FILE__, __LINE__,
		         "==============BayesianLR::multiplier(0)==1.0==============",
		         fabsf(m - 1.0f) < 1e-6f);

		float m50 = cfg.multiplier(50);
		G_assert(__FILE__, __LINE__,
		         "==============BayesianLR::multiplier(50)==1.0==============",
		         fabsf(m50 - 1.0f) < 1e-6f);

		printf("    BAYESIAN enum = %d, multiplier(0) = %f, multiplier(50) = %f\n",
		       static_cast<int>(cfg.type), m, m50);
	}

	// ---- Sub-test 2: BayesianLRConfig defaults ----
	printf("  Sub-test 2: BayesianLRConfig defaults\n");
	{
		glades::BayesianLRConfig blr;
		G_assert(__FILE__, __LINE__,
		         "==============BayesianLRConfig::windowEpochs==10==============",
		         blr.windowEpochs == 10);
		G_assert(__FILE__, __LINE__,
		         "==============BayesianLRConfig::minLR==1e-6==============",
		         fabsf(blr.minLR - 1e-6f) < 1e-10f);
		G_assert(__FILE__, __LINE__,
		         "==============BayesianLRConfig::maxLR==0.1==============",
		         fabsf(blr.maxLR - 0.1f) < 1e-6f);
		printf("    windowEpochs = %d, minLR = %e, maxLR = %f\n",
		       blr.windowEpochs, blr.minLR, blr.maxLR);
	}

	// ---- Sub-test 3: TrainingConfig has bayesianLR field ----
	printf("  Sub-test 3: TrainingConfig has bayesianLR field\n");
	{
		glades::TrainingConfig cfg;
		G_assert(__FILE__, __LINE__,
		         "==============TrainingConfig::bayesianLR.windowEpochs==10==============",
		         cfg.bayesianLR.windowEpochs == 10);
		G_assert(__FILE__, __LINE__,
		         "==============TrainingConfig::bayesianLR.minLR==1e-6==============",
		         fabsf(cfg.bayesianLR.minLR - 1e-6f) < 1e-10f);
		G_assert(__FILE__, __LINE__,
		         "==============TrainingConfig::bayesianLR.maxLR==0.1==============",
		         fabsf(cfg.bayesianLR.maxLR - 0.1f) < 1e-6f);
	}

	printf("=== All BayesianLRSchedule Unit Tests PASSED ===\n");
}

void HyperparameterTunerFullLoopTest()
{
	printf("=== HyperparameterTuner Full Loop Test ===\n");

	// ---- Build a simple DFF regression dataset (y = 2*x1 + x2) ----
	glades::NumberInput* trainDI = new glades::NumberInput();
	trainDI->trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
	trainDI->trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
	// Row 0: [0,0] -> 0
	trainDI->trainMatrix[0][0] = 0.0f; trainDI->trainMatrix[0][1] = 0.0f;
	trainDI->trainExpectedMatrix[0][0] = 0.0f;
	// Row 1: [1,0] -> 2
	trainDI->trainMatrix[1][0] = 1.0f; trainDI->trainMatrix[1][1] = 0.0f;
	trainDI->trainExpectedMatrix[1][0] = 2.0f;
	// Row 2: [0,1] -> 1
	trainDI->trainMatrix[2][0] = 0.0f; trainDI->trainMatrix[2][1] = 1.0f;
	trainDI->trainExpectedMatrix[2][0] = 1.0f;
	// Row 3: [1,1] -> 3
	trainDI->trainMatrix[3][0] = 1.0f; trainDI->trainMatrix[3][1] = 1.0f;
	trainDI->trainExpectedMatrix[3][0] = 3.0f;

	// Use same data for validation
	trainDI->testMatrix = trainDI->trainMatrix;
	trainDI->testExpectedMatrix = trainDI->trainExpectedMatrix;

	// ---- Build the template network ----
	auto in = shmea::make_gpointer<glades::InputLayerInfo>(
	    /*batchSize*/ 4,
	    /*learningRate*/ 0.1f,
	    /*momentumFactor*/ 0.0f,
	    /*weightDecay1*/ 0.0f,
	    /*weightDecay2*/ 0.0f,
	    /*pDropout*/ 0.0f,
	    /*activationType*/ glades::GMath::LINEAR,
	    /*activationParam*/ 1.0f);
	std::vector<shmea::GPointer<glades::HiddenLayerInfo>> hidden;
	hidden.push_back(shmea::make_gpointer<glades::HiddenLayerInfo>(
	    /*size*/ 4,
	    /*learningRate*/ 0.1f,
	    /*momentumFactor*/ 0.0f,
	    /*weightDecay1*/ 0.0f,
	    /*weightDecay2*/ 0.0f,
	    /*pDropout*/ 0.0f,
	    /*activationType*/ glades::GMath::LINEAR,
	    /*activationParam*/ 1.0f));
	auto out = shmea::make_gpointer<glades::OutputLayerInfo>(1, glades::OutputLayerInfo::REGRESSION);
	glades::NNInfo* info = new glades::NNInfo("ut_hp_tuner_full", in, hidden, out);

	glades::NNetwork templateNet(info, glades::NNetwork::TYPE_DFF);
	templateNet.setSeed(42u);
	templateNet.getTerminatorMutable().setEpoch(50);
	templateNet.getTerminatorMutable().setAccuracy(0);

	// ---- Build a simple 2-param search space ----
	glades::SearchSpace space;
	{
		glades::HyperParameter hp;
		hp.name = "learningRate";
		hp.type = glades::HyperParameter::CONTINUOUS;
		hp.low = 0.01f;
		hp.high = 0.5f;
		hp.logScale = true;
		space.params.push_back(hp);
	}
	{
		glades::HyperParameter hp;
		hp.name = "globalGradClipNorm";
		hp.type = glades::HyperParameter::CONTINUOUS;
		hp.low = 0.0f;
		hp.high = 10.0f;
		hp.logScale = false;
		space.params.push_back(hp);
	}

	// ---- Run 3 trials, 50 epochs each ----
	printf("  Running 3 trials, 50 epochs each...\n");
	glades::HyperparameterTuner tuner(space, 3, 50);

	glades::TrainingConfig bestConfig = tuner.optimize(templateNet, trainDI, trainDI);

	// ---- Verify results ----
	printf("  Sub-test 1: 3 trials completed\n");
	G_assert(__FILE__, __LINE__,
	         "==============HPTunerFull::trials size==3==============",
	         tuner.getTrials().size() == 3);

	printf("  Sub-test 2: all trials completed\n");
	for (unsigned int i = 0; i < tuner.getTrials().size(); ++i)
	{
		G_assert(__FILE__, __LINE__,
		         "==============HPTunerFull::trial completed==============",
		         tuner.getTrials()[i].completed);
	}

	printf("  Sub-test 3: best score is finite\n");
	float bestScore = tuner.getBestScore();
	printf("    bestScore = %f\n", bestScore);
	G_assert(__FILE__, __LINE__,
	         "==============HPTunerFull::bestScore finite==============",
	         std::isfinite(bestScore));

	printf("  Sub-test 4: best score is non-negative\n");
	G_assert(__FILE__, __LINE__,
	         "==============HPTunerFull::bestScore >= 0==============",
	         bestScore >= 0.0f);

	// Print trial summaries
	for (unsigned int i = 0; i < tuner.getTrials().size(); ++i)
	{
		const glades::HyperparameterTuner::Trial& t = tuner.getTrials()[i];
		printf("    Trial %d: score=%f, params=[", t.id, t.score);
		for (unsigned int j = 0; j < t.rawParams.size(); ++j)
		{
			if (j > 0) printf(", ");
			printf("%f", t.rawParams[j]);
		}
		printf("]\n");
	}

	delete trainDI;
	delete info;

	printf("=== All HyperparameterTuner Full Loop Tests PASSED ===\n");
}
