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
#include "nn-cv-test.h"
#include "../../unit-test.h"
#include "Backend/Database/GList.h"
#include "Backend/Database/GTable.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/cross_validation.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"
#include <cmath>

// === This is the primary unit testing function:
// void G_assert(const char* fileName, int lineNo, const char* failureMsg, bool expr)

void NNCVUnitTestValidation()
{
	printf("============================================================\n");
	printf("NN Cross-Validation Unit Test Suite (modern)\n");
	printf("============================================================\n");

	// Build a tiny in-memory classification dataset:
	// - 2 numeric features
	// - 1 string label output column (binary)
	shmea::GVector<shmea::GString> headers;
	headers.push_back("x1");
	headers.push_back("x2");
	headers.push_back("label");
	shmea::GTable tbl(',', headers);
	tbl.toggleOutput(2u); // "label"

	for (int i = 0; i < 10; ++i)
	{
		const float x1 = (i < 5) ? -1.0f : 1.0f;
		const float x2 = static_cast<float>(i);
		const char* lab = (i < 5) ? "A" : "B";
		shmea::GList row;
		row.addFloat(x1);
		row.addFloat(x2);
		row.addString(lab);
		tbl.addRow(row);
	}

	// Simple DFF binary classifier (no hidden layers).
	glades::InputLayerInfo* in = new glades::InputLayerInfo(
	    /*batchSize*/ 10,
	    /*learningRate*/ 0.20f,
	    /*momentumFactor*/ 0.0f,
	    /*weightDecay1*/ 0.0f,
	    /*weightDecay2*/ 0.0f,
	    /*pDropout*/ 0.0f,
	    /*activationType*/ glades::GMath::LINEAR,
	    /*activationParam*/ 1.0f);
	std::vector<glades::HiddenLayerInfo*> hidden;
	glades::OutputLayerInfo* out = new glades::OutputLayerInfo(2, glades::OutputLayerInfo::CLASSIFICATION);
	glades::NNInfo* info = new glades::NNInfo("ut_cv_bin", in, hidden, out);

	glades::NNetwork net1(info, glades::NNetwork::TYPE_DFF);
	glades::NNetwork net2(info, glades::NNetwork::TYPE_DFF);
	net1.setSeed(123u);
	net2.setSeed(456u);
	net1.getTerminatorMutable().setEpoch(3);
	net2.getTerminatorMutable().setEpoch(3);

	std::vector<glades::NNetwork*> models;
	models.push_back(&net1);
	models.push_back(&net2);

	printf("-----------------------------------\n");
	printf("CV Test A (k-fold deterministic API)\n");
	printf("-----------------------------------\n");
	{
		glades::CrossValidationConfig cfg;
		cfg.kFolds = 5u;
		cfg.shuffle = true;
		cfg.seed = 2026u;
		cfg.timeSeries = false;
		cfg.stratify = true;
		cfg.standardizeFlag = glades::GMath::ZSCORE;

		glades::CrossValidationResults r1;
		const glades::NNetworkStatus st1 = glades::crossValidateTableCSV(tbl, models, cfg, &r1);
		G_assert(__FILE__, __LINE__, "==============NNCV::A Status Failed==============", st1.ok());
		G_assert(__FILE__, __LINE__, "==============NNCV::A MeanSizeMismatch Failed==============", r1.meanTestAccuracy.size() == models.size());
		G_assert(__FILE__, __LINE__, "==============NNCV::A FoldOuterSizeMismatch Failed==============", r1.foldTestAccuracy.size() == models.size());
		G_assert(__FILE__, __LINE__, "==============NNCV::A FoldsUsedNonZero Failed==============", r1.foldsUsed > 0u);
		for (unsigned int m = 0; m < r1.meanTestAccuracy.size(); ++m)
		{
			const float a = r1.meanTestAccuracy[m];
			G_assert(__FILE__, __LINE__, "==============NNCV::A MeanNonFinite Failed==============", std::isfinite(a));
			G_assert(__FILE__, __LINE__, "==============NNCV::A MeanRange Failed==============", (a >= 0.0f) && (a <= 100.0f));
			G_assert(__FILE__, __LINE__, "==============NNCV::A FoldCount Failed==============", r1.foldTestAccuracy[m].size() == r1.foldsUsed);
		}

		// Determinism: same config => identical results.
		glades::CrossValidationResults r2;
		const glades::NNetworkStatus st2 = glades::crossValidateTableCSV(tbl, models, cfg, &r2);
		G_assert(__FILE__, __LINE__, "==============NNCV::A2 Status Failed==============", st2.ok());
		G_assert(__FILE__, __LINE__, "==============NNCV::A2 Determinism MeanSize Failed==============", r2.meanTestAccuracy.size() == r1.meanTestAccuracy.size());
		for (unsigned int m = 0; m < r1.meanTestAccuracy.size(); ++m)
		{
			G_assert(__FILE__, __LINE__, "==============NNCV::A2 Determinism MeanMismatch Failed==============",
			         fabs(r2.meanTestAccuracy[m] - r1.meanTestAccuracy[m]) < 1e-6f);
		}
	}

	printf("-----------------------------------\n");
	printf("CV Test B (walk-forward folds skip empty-train)\n");
	printf("-----------------------------------\n");
	{
		glades::CrossValidationConfig cfg;
		cfg.kFolds = 5u;
		cfg.timeSeries = true;
		cfg.shuffle = false;
		cfg.stratify = false;
		cfg.seed = 7u;
		cfg.standardizeFlag = glades::GMath::NONE;

		glades::CrossValidationResults r;
		const glades::NNetworkStatus st = glades::crossValidateTableCSV(tbl, models, cfg, &r);
		G_assert(__FILE__, __LINE__, "==============NNCV::B Status Failed==============", st.ok());
		// For N=10, k=5, the first fold has empty train (start=0) and is skipped -> 4 usable folds.
		G_assert(__FILE__, __LINE__, "==============NNCV::B FoldsUsed Failed==============", r.foldsUsed == 4u);
	}

	delete info;

	printf("\n============================================================\n");
}

