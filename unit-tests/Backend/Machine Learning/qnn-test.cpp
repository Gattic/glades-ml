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

#include "qnn-test.h"
#include "../../unit-test.h"

#include "../../../Backend/Machine Learning/main.h"
#include "../../../Backend/Machine Learning/DataObjects/NumberInput.h"
#include "../../../Backend/Machine Learning/DataObjects/QuaternionInput.h"
#include "../../../Backend/Machine Learning/GMath/gmath.h"
#include "../../../Backend/Machine Learning/GMath/quaternion.h"
#include "../../../Backend/Machine Learning/Networks/network.h"
#include "../../../Backend/Machine Learning/Networks/training_callbacks.h"
#include "../../../Backend/Machine Learning/State/Terminator.h"
#include "../../../Backend/Machine Learning/Structure/hiddenlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/inputlayerinfo.h"
#include "../../../Backend/Machine Learning/Structure/nninfo.h"
#include "../../../Backend/Machine Learning/Structure/outputlayerinfo.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

static bool near_f(float a, float b, float tol)
{
	return fabsf(a - b) <= tol;
}

static bool near_q(const glades::Quaternion& a, const glades::Quaternion& b, float tol)
{
	return near_f(a.r, b.r, tol) &&
	       near_f(a.i, b.i, tol) &&
	       near_f(a.j, b.j, tol) &&
	       near_f(a.k, b.k, tol);
}

struct CaptureLastEpochCb : public glades::ITrainingCallbacks
{
	glades::NNetworkEpochMetrics first;
	glades::NNetworkEpochMetrics last;
	bool saw;
	unsigned int firstPerfectEpoch;

	CaptureLastEpochCb() : first(), last(), saw(false), firstPerfectEpoch(0u) {}
	virtual void onRunStart(const glades::NNetwork&, int) {}
	virtual bool onEpochEnd(const glades::NNetwork&, const glades::NNetworkEpochMetrics& m)
	{
		if (!saw)
			first = m;
		last = m;
		saw = true;
		if (firstPerfectEpoch == 0u && m.totalAccuracy >= 99.9f)
			firstPerfectEpoch = static_cast<unsigned int>(m.epoch);
		return false;
	}
	virtual void onRunEnd(const glades::NNetwork&, int) {}
};

struct BenchmarkSummaryRow
{
	std::string label;
	std::string arch;
	unsigned int seeds;
	unsigned int epochs;
	double meanTrainTop1Pct;
	double meanTestTop1Pct;
	double meanTestMacroAccPct;
	double meanTestLoss;
};

static std::string current_working_dir()
{
	char buf[4096];
	if (::getcwd(buf, sizeof(buf)) != NULL)
		return std::string(buf);
	return std::string();
}

static void remove_model_dir_best_effort(const std::string& modelName)
{
	const std::string dir = "database/models/" + modelName;
	(void)::remove((dir + "/manifest.txt").c_str());
	(void)::remove((dir + "/nninfo.csv").c_str());
	(void)::remove((dir + "/weights.bin").c_str());
	(void)::rmdir(dir.c_str());
}

static void test_quaternion_algebra()
{
	using glades::Quaternion;

	printf("-----------------------------------\n");
	printf("QNN Phase 0a: Quaternion Algebra Identities\n");
	printf("-----------------------------------\n");

	ASSERT("Quaternion is 16 bytes contiguous", sizeof(Quaternion) == 16u);

	const Quaternion one(1.0f, 0.0f, 0.0f, 0.0f);
	const Quaternion qi(0.0f, 1.0f, 0.0f, 0.0f);
	const Quaternion qj(0.0f, 0.0f, 1.0f, 0.0f);
	const Quaternion qk(0.0f, 0.0f, 0.0f, 1.0f);
	const Quaternion negOne(-1.0f, 0.0f, 0.0f, 0.0f);

	// Basis squares and i*j*k = -1
	ASSERT("i^2 == -1", near_q(Quaternion::mul(qi, qi), negOne, 1e-6f));
	ASSERT("j^2 == -1", near_q(Quaternion::mul(qj, qj), negOne, 1e-6f));
	ASSERT("k^2 == -1", near_q(Quaternion::mul(qk, qk), negOne, 1e-6f));
	ASSERT("i*j*k == -1", near_q(Quaternion::mul(Quaternion::mul(qi, qj), qk), negOne, 1e-6f));

	// Anti-commutative basis cross products
	ASSERT("i*j == k", near_q(Quaternion::mul(qi, qj), qk, 1e-6f));
	ASSERT("j*i == -k", near_q(Quaternion::mul(qj, qi), -qk, 1e-6f));
	ASSERT("j*k == i", near_q(Quaternion::mul(qj, qk), qi, 1e-6f));
	ASSERT("k*j == -i", near_q(Quaternion::mul(qk, qj), -qi, 1e-6f));
	ASSERT("k*i == j", near_q(Quaternion::mul(qk, qi), qj, 1e-6f));
	ASSERT("i*k == -j", near_q(Quaternion::mul(qi, qk), -qj, 1e-6f));

	const Quaternion a(1.25f, -0.75f, 2.1f, -0.4f);
	const Quaternion b(-0.6f, 1.8f, -0.9f, 1.3f);
	const Quaternion c(0.35f, 0.95f, -1.45f, 0.85f);

	// Non-commutativity, associativity, norm multiplicativity
	const Quaternion ab = Quaternion::mul(a, b);
	const Quaternion ba = Quaternion::mul(b, a);
	ASSERT("Hamilton product is non-commutative", !near_q(ab, ba, 1e-3f));
	ASSERT("Hamilton product is associative",
	       near_q(Quaternion::mul(ab, c), Quaternion::mul(a, Quaternion::mul(b, c)), 1e-5f));
	ASSERT("Norm is multiplicative", near_f(ab.norm(), a.norm() * b.norm(), 1e-5f));
	ASSERT("a * 1 == a", near_q(Quaternion::mul(a, one), a, 1e-6f));
	ASSERT("1 * a == a", near_q(Quaternion::mul(one, a), a, 1e-6f));

	// Conjugate identities
	ASSERT("(a*)* == a", near_q(a.conj().conj(), a, 1e-6f));
	ASSERT("(a*b)* == b* * a*", near_q(ab.conj(), Quaternion::mul(b.conj(), a.conj()), 1e-5f));
	ASSERT("a * a* == (||a||^2, 0, 0, 0)",
	       near_q(Quaternion::mul(a, a.conj()), Quaternion(a.normSq(), 0.0f, 0.0f, 0.0f), 1e-5f));

	// Specialized backprop products & adjoint inner-product identities
	ASSERT("mulConjRight(a, b) == mul(a, b.conj())",
	       near_q(Quaternion::mulConjRight(a, b), Quaternion::mul(a, b.conj()), 1e-6f));
	ASSERT("mulConjLeft(a, b) == mul(a.conj(), b)",
	       near_q(Quaternion::mulConjLeft(a, b), Quaternion::mul(a.conj(), b), 1e-6f));
	ASSERT("<w (x) a, d> == <w, d (x) a*>",
	       near_f(Quaternion::dot(Quaternion::mul(a, b), c),
	              Quaternion::dot(a, Quaternion::mulConjRight(c, b)), 1e-5f));
	ASSERT("<w (x) a, d> == <a, w* (x) d>",
	       near_f(Quaternion::dot(Quaternion::mul(a, b), c),
	              Quaternion::dot(b, Quaternion::mulConjLeft(a, c)), 1e-5f));

	// Finiteness checks
	ASSERT("Finite quaternion passes isFinite()", a.isFinite());
	const float qnan = std::numeric_limits<float>::quiet_NaN();
	const float pinf = std::numeric_limits<float>::infinity();
	ASSERT("NaN quaternion fails isFinite()", !Quaternion(qnan, 0.0f, 0.0f, 0.0f).isFinite());
	ASSERT("Inf quaternion fails isFinite()", !Quaternion(0.0f, 0.0f, pinf, 0.0f).isFinite());

	printf("[PASS] Quaternion algebra identities verified.\n");
}

static glades::NNInfo* make_qnn_info(const char* name,
                                     int batchSize,
                                     float lr,
                                     float momentum,
                                     float wd2,
                                     const std::vector<unsigned int>& hiddenQuats,
                                     int hiddenAct,
                                     int outSize,
                                     int outType,
                                     int outAct)
{
	shmea::GPointer<glades::InputLayerInfo> in(new glades::InputLayerInfo(
	    batchSize, lr, momentum, 0.0f, wd2, 0.0f, hiddenAct, 1.0f));

	std::vector< shmea::GPointer<glades::HiddenLayerInfo> > hidden;
	for (size_t i = 0u; i < hiddenQuats.size(); ++i)
	{
		const int actForTransition = (i + 1u == hiddenQuats.size()) ? outAct : hiddenAct;
		hidden.push_back(shmea::GPointer<glades::HiddenLayerInfo>(new glades::HiddenLayerInfo(
		    static_cast<int>(hiddenQuats[i]), lr, momentum, 0.0f, wd2, 0.0f, actForTransition, 1.0f)));
	}

	shmea::GPointer<glades::OutputLayerInfo> out(new glades::OutputLayerInfo(outSize, outType));
	return new glades::NNInfo(shmea::GString(name), in, hidden, out);
}

static void eval_split_predictions(glades::NNetwork& net,
                                   const glades::NumberInput& di,
                                   bool useTrainSplit,
                                   std::vector< std::vector<float> >& outPreds,
                                   unsigned int& outCorrect,
                                   unsigned int& outTotal,
                                   std::vector<unsigned int>* perClassCorrect,
                                   std::vector<unsigned int>* perClassTotal)
{
	const unsigned int nRows = useTrainSplit ? di.getTrainSize() : di.getTestSize();
	const unsigned int outSize = net.getNNInfo()->getOutputLayerSize();
	outPreds.assign(nRows, std::vector<float>(outSize, 0.0f));
	outCorrect = 0u;
	outTotal = nRows;
	if (perClassCorrect)
		perClassCorrect->assign(outSize > 1u ? outSize : 2u, 0u);
	if (perClassTotal)
		perClassTotal->assign(outSize > 1u ? outSize : 2u, 0u);

	for (unsigned int r = 0u; r < nRows; ++r)
	{
		glades::NumberInput single;
		single.testMatrix = shmea::GMatrix(1, shmea::GVector<float>(di.getFeatureCount(), 0.0f));
		single.testExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(outSize, 0.0f));
		const float* x = NULL;
		unsigned int xN = 0u;
		const float* y = NULL;
		unsigned int yN = 0u;
		if (useTrainSplit)
		{
			di.getTrainRowView(r, x, xN);
			di.getTrainExpectedRowView(r, y, yN);
		}
		else
		{
			di.getTestRowView(r, x, xN);
			di.getTestExpectedRowView(r, y, yN);
		}
		for (unsigned int c = 0u; c < xN; ++c)
			single.testMatrix[0][c] = x[c];
		for (unsigned int c = 0u; c < yN; ++c)
			single.testExpectedMatrix[0][c] = y[c];
		single.trainMatrix = single.testMatrix;
		single.trainExpectedMatrix = single.testExpectedMatrix;
		single.loaded = true;

		glades::NNetworkStatus st = net.test(&single);
		ASSERT("Single-row QNN evaluation succeeds", st.ok());
		shmea::GList res = net.getResults();
		ASSERT("Single-row QNN results size matches 2*outSize", res.size() == outSize * 2u);
		for (unsigned int k = 0u; k < outSize; ++k)
			outPreds[r][k] = res.getFloat(k * 2u + 1u);

		unsigned int expCls = 0u;
		unsigned int predCls = 0u;
		if (outSize > 1u)
		{
			expCls = static_cast<unsigned int>(glades::GMath::argmax(y, outSize));
			predCls = static_cast<unsigned int>(glades::GMath::argmax(&outPreds[r][0], outSize));
		}
		else
		{
			expCls = (y && y[0] >= 0.5f) ? 1u : 0u;
			predCls = (outPreds[r][0] >= 0.5f) ? 1u : 0u;
		}
		if (perClassTotal && expCls < perClassTotal->size())
			++(*perClassTotal)[expCls];
		if (expCls == predCls)
		{
			++outCorrect;
			if (perClassCorrect && expCls < perClassCorrect->size())
				++(*perClassCorrect)[expCls];
		}
	}
}

static void eval_single_row_predictions(glades::NNetwork& net,
                                        const glades::NumberInput& di,
                                        std::vector< std::vector<float> >& outPreds)
{
	unsigned int correct = 0u;
	unsigned int total = 0u;
	eval_split_predictions(net, di, false, outPreds, correct, total, NULL, NULL);
}

static void test_quaternion_input_loaders()
{
	printf("-----------------------------------\n");
	printf("QNN Loader Verification: QuaternionInput (XOR & Iris)\n");
	printf("-----------------------------------\n");

	// 1) XOR loader: 2 features -> 1 quaternion (x, y, 0, 0), 1 target -> 1 quaternion (t, 0, 0, 0)
	{
		glades::QuaternionInput xorQi;
		ASSERT("QuaternionInput::loadXOR succeeds",
		       xorQi.loadXOR("datasets/xorgate.csv",
		                     glades::QUAT_PACK_GROUPS_OF_4,
		                     glades::QUAT_READOUT_REAL_PART));
		ASSERT("XOR train size is 4", xorQi.getTrainSize() == 4u);
		ASSERT("XOR real feature count is 2", xorQi.getFeatureCount() == 2u);
		ASSERT("XOR quaternion input count is 1", xorQi.getQuatInputCount() == 1u);
		ASSERT("XOR quaternion output count is 1", xorQi.getQuatOutputCount() == 1u);

		for (unsigned int r = 0u; r < 4u; ++r)
		{
			const glades::Quaternion* qIn = NULL;
			unsigned int qInN = 0u;
			const glades::Quaternion* qOut = NULL;
			unsigned int qOutN = 0u;
			ASSERT("XOR train quat input row valid", xorQi.getTrainQuatInputRow(r, qIn, qInN));
			ASSERT("XOR train quat target row valid", xorQi.getTrainQuatTargetRow(r, qOut, qOutN));
			ASSERT("XOR qInN == 1", qInN == 1u);
			ASSERT("XOR qOutN == 1", qOutN == 1u);

			const float* x = NULL;
			unsigned int xN = 0u;
			const float* y = NULL;
			unsigned int yN = 0u;
			xorQi.getTrainRowView(r, x, xN);
			xorQi.getTrainExpectedRowView(r, y, yN);
			ASSERT("XOR input quaternion packs (x0, x1, 0, 0)",
			       near_q(qIn[0], glades::Quaternion(x[0], x[1], 0.0f, 0.0f), 1e-6f));
			ASSERT("XOR target quaternion stores (y0, 0, 0, 0)",
			       near_q(qOut[0], glades::Quaternion(y[0], 0.0f, 0.0f, 0.0f), 1e-6f));
		}
	}

	// 2) Iris loader in default PACK_GROUPS_OF_4 + READOUT_REAL_PART:
	//    4 features (SL, SW, PL, PW) -> 1 quaternion (SL, SW, PL, PW)
	//    3 species classes -> 3 output quaternions [(y0,0,0,0), (y1,0,0,0), (y2,0,0,0)]
	{
		glades::NumberInput::TrainTestSplitConfig splitCfg;
		splitCfg.testFraction = 0.2f;
		splitCfg.shuffle = true;
		splitCfg.stratify = true;
		splitCfg.seed = 42ULL;

		glades::QuaternionInput irisReal;
		ASSERT("QuaternionInput::loadIris (PACK4 + REAL_PART) succeeds",
		       irisReal.loadIris("datasets/iris.data",
		                         splitCfg,
		                         glades::GMath::ZSCORE,
		                         glades::QUAT_PACK_GROUPS_OF_4,
		                         glades::QUAT_READOUT_REAL_PART));
		ASSERT("Iris train size is 120", irisReal.getTrainSize() == 120u);
		ASSERT("Iris test size is 30", irisReal.getTestSize() == 30u);
		ASSERT("Iris feature count is 4", irisReal.getFeatureCount() == 4u);
		ASSERT("Iris real output count is 3", irisReal.getRealOutputCount() == 3u);
		ASSERT("Iris quat input count is 1 (4 features -> 1 quat)", irisReal.getQuatInputCount() == 1u);
		ASSERT("Iris quat output count is 3 (REAL_PART)", irisReal.getQuatOutputCount() == 3u);
		ASSERT("Iris class labels has 3 species", irisReal.getClassLabels().size() == 3u);

		// Verify stratified class counts (40 train / 10 test per class) & exact quaternion mapping.
		unsigned int trainClassCounts[3] = { 0u, 0u, 0u };
		double laneSum[4] = { 0.0, 0.0, 0.0, 0.0 };
		double laneSumSq[4] = { 0.0, 0.0, 0.0, 0.0 };
		for (unsigned int r = 0u; r < irisReal.getTrainSize(); ++r)
		{
			const glades::Quaternion* qIn = NULL;
			unsigned int qInN = 0u;
			const glades::Quaternion* qOut = NULL;
			unsigned int qOutN = 0u;
			ASSERT("Iris train quat input row valid", irisReal.getTrainQuatInputRow(r, qIn, qInN));
			ASSERT("Iris train quat target row valid", irisReal.getTrainQuatTargetRow(r, qOut, qOutN));
			ASSERT("Iris train qInN == 1", qInN == 1u);
			ASSERT("Iris train qOutN == 3", qOutN == 3u);

			const float* x = NULL;
			unsigned int xN = 0u;
			const float* y = NULL;
			unsigned int yN = 0u;
			irisReal.getTrainRowView(r, x, xN);
			irisReal.getTrainExpectedRowView(r, y, yN);
			ASSERT("Iris input quaternion equals (SL, SW, PL, PW)",
			       near_q(qIn[0], glades::Quaternion(x[0], x[1], x[2], x[3]), 1e-6f));

			for (unsigned int lane = 0u; lane < 4u; ++lane)
			{
				const double v = static_cast<double>(qIn[0].component(lane));
				laneSum[lane] += v;
				laneSumSq[lane] += v * v;
			}

			float decoded[3] = { 0.0f, 0.0f, 0.0f };
			glades::decodeQuaternionsToReal(qOut, qOutN, glades::QUAT_READOUT_REAL_PART, decoded, 3u);
			for (unsigned int c = 0u; c < 3u; ++c)
			{
				ASSERT("Iris REAL_PART target quaternion matches one-hot class",
				       near_q(qOut[c], glades::Quaternion(y[c], 0.0f, 0.0f, 0.0f), 1e-6f));
				ASSERT("Iris REAL_PART decode round-trips one-hot", near_f(decoded[c], y[c], 1e-6f));
			}
			const int cls = glades::GMath::argmax(y, 3u);
			ASSERT("Iris train class index in [0,2]", cls >= 0 && cls < 3);
			++trainClassCounts[cls];
		}
		for (unsigned int c = 0u; c < 3u; ++c)
			ASSERT("Iris train split is stratified (40 per class)", trainClassCounts[c] == 40u);
		for (unsigned int lane = 0u; lane < 4u; ++lane)
		{
			const double mean = laneSum[lane] / 120.0;
			const double var = (laneSumSq[lane] / 120.0) - (mean * mean);
			ASSERT("Iris train quaternion lane z-score mean ~ 0", fabs(mean) < 1e-4);
			ASSERT("Iris train quaternion lane z-score std ~ 1", fabs(sqrt(var) - 1.0) < 2e-2);
		}

		unsigned int testClassCounts[3] = { 0u, 0u, 0u };
		for (unsigned int r = 0u; r < irisReal.getTestSize(); ++r)
		{
			const float* y = NULL;
			unsigned int yN = 0u;
			irisReal.getTestExpectedRowView(r, y, yN);
			const int cls = glades::GMath::argmax(y, 3u);
			ASSERT("Iris test class index in [0,2]", cls >= 0 && cls < 3);
			++testClassCounts[cls];
		}
		for (unsigned int c = 0u; c < 3u; ++c)
			ASSERT("Iris test split is stratified (10 per class)", testClassCounts[c] == 10u);

		// 3) Iris loader in compact READOUT_SPLIT_COMPONENTS:
		//    3 classes -> 1 output quaternion (y0, y1, y2, 0)
		glades::QuaternionInput irisSplit;
		ASSERT("QuaternionInput::loadIris (PACK4 + SPLIT_COMPONENTS) succeeds",
		       irisSplit.loadIris("datasets/iris.data",
		                          splitCfg,
		                          glades::GMath::ZSCORE,
		                          glades::QUAT_PACK_GROUPS_OF_4,
		                          glades::QUAT_READOUT_SPLIT_COMPONENTS));
		ASSERT("Iris SPLIT_COMPONENTS quat output count is 1", irisSplit.getQuatOutputCount() == 1u);
		for (unsigned int r = 0u; r < irisSplit.getTestSize(); ++r)
		{
			const glades::Quaternion* qOut = NULL;
			unsigned int qOutN = 0u;
			ASSERT("Iris split test quat target row valid", irisSplit.getTestQuatTargetRow(r, qOut, qOutN));
			ASSERT("Iris split test qOutN == 1", qOutN == 1u);
			const float* y = NULL;
			unsigned int yN = 0u;
			irisSplit.getTestExpectedRowView(r, y, yN);
			ASSERT("Iris SPLIT_COMPONENTS packs 3 classes into (r, i, j, 0)",
			       near_q(qOut[0], glades::Quaternion(y[0], y[1], y[2], 0.0f), 1e-6f));
			float decoded[3] = { 0.0f, 0.0f, 0.0f };
			glades::decodeQuaternionsToReal(qOut, qOutN, glades::QUAT_READOUT_SPLIT_COMPONENTS, decoded, 3u);
			for (unsigned int c = 0u; c < 3u; ++c)
				ASSERT("Iris SPLIT_COMPONENTS decode round-trips one-hot", near_f(decoded[c], y[c], 1e-6f));
		}

		// 4) Iris loader in ONE_QUAT_PER_FEATURE mode: 4 features -> 4 real-lane quaternions
		glades::QuaternionInput irisPerFeat;
		ASSERT("QuaternionInput::loadIris (ONE_PER_FEATURE + REAL_PART) succeeds",
		       irisPerFeat.loadIris("datasets/iris.data",
		                            splitCfg,
		                            glades::GMath::ZSCORE,
		                            glades::QUAT_ONE_PER_FEATURE,
		                            glades::QUAT_READOUT_REAL_PART));
		ASSERT("Iris ONE_PER_FEATURE quat input count is 4", irisPerFeat.getQuatInputCount() == 4u);
		{
			const glades::Quaternion* qIn = NULL;
			unsigned int qInN = 0u;
			const float* x = NULL;
			unsigned int xN = 0u;
			ASSERT("Iris per-feature row 0 valid", irisPerFeat.getTrainQuatInputRow(0u, qIn, qInN));
			irisPerFeat.getTrainRowView(0u, x, xN);
			ASSERT("Iris per-feature qInN == 4", qInN == 4u);
			for (unsigned int f = 0u; f < 4u; ++f)
				ASSERT("Iris per-feature maps x_f -> (x_f, 0, 0, 0)",
				       near_q(qIn[f], glades::Quaternion(x[f], 0.0f, 0.0f, 0.0f), 1e-6f));
		}
	}

	printf("[PASS] QuaternionInput XOR & Iris loaders verified.\n");
}

static void test_qnn_xor(std::vector<BenchmarkSummaryRow>& summary)
{
	printf("-----------------------------------\n");
	printf("QNN Phase 1: XOR Gate Convergence\n");
	printf("-----------------------------------\n");

	glades::QuaternionInput xorDi;
	ASSERT("Load datasets/xorgate.csv via QuaternionInput::loadXOR succeeds",
	       xorDi.loadXOR("datasets/xorgate.csv",
	                     glades::QUAT_PACK_GROUPS_OF_4,
	                     glades::QUAT_READOUT_REAL_PART));
	ASSERT("XOR train size is 4", xorDi.getTrainSize() == 4u);
	ASSERT("XOR feature count is 2", xorDi.getFeatureCount() == 2u);
	ASSERT("XOR quat input count is 1", xorDi.getQuatInputCount() == 1u);
	ASSERT("XOR quat output count is 1", xorDi.getQuatOutputCount() == 1u);

	// 1a. Primary architecture: Input(2) -> 1 quat -> Hidden(4 quats, TANH) -> Output(1 quat, SIGMOID)
	// Full-batch AdamW across 5 deterministic seeds.
	const uint64_t seeds[5] = { 101ULL, 202ULL, 303ULL, 404ULL, 505ULL };
	std::vector<unsigned int> hidden4;
	hidden4.push_back(4u);

	double adamTop1Sum = 0.0;
	double adamMacroSum = 0.0;
	double adamLossSum = 0.0;

	for (unsigned int s = 0u; s < 5u; ++s)
	{
		glades::NNInfo* info = make_qnn_info(
		    "qnn_xor_adamw", /*batchSize*/ 4, /*lr*/ 0.05f, /*momentum*/ 0.0f, /*wd2*/ 0.0f,
		    hidden4, glades::GMath::TANH, /*outSize*/ 1, glades::OutputLayerInfo::CLASSIFICATION, glades::GMath::SIGMOID);

		glades::NNetwork net(info, glades::NNetwork::TYPE_QNN);
		delete info;

		net.setSeed(seeds[s]);
		glades::TrainingConfig cfg = net.getTrainingConfig();
		cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
		cfg.qnn.inputPacking = glades::QuaternionConfig::PACK_GROUPS_OF_4;
		cfg.qnn.outputReadout = glades::QuaternionConfig::READOUT_REAL_PART;
		ASSERT("XOR setTrainingConfig succeeds", net.setTrainingConfig(cfg).ok());

		glades::Terminator term;
		term.setEpoch(600);
		ASSERT("XOR setTerminator succeeds", net.setTerminator(term).ok());

		CaptureLastEpochCb cb;
		glades::NNetworkStatus st = net.train(&xorDi, &cb);
		ASSERT("XOR AdamW train succeeds", st.ok());
		ASSERT("XOR AdamW observed epoch metrics", cb.saw);

		CaptureLastEpochCb evalCb;
		st = net.test(&xorDi, &evalCb);
		ASSERT("XOR AdamW test succeeds", st.ok());

		std::vector< std::vector<float> > preds;
		unsigned int correct = 0u, total = 0u;
		eval_split_predictions(net, xorDi, false, preds, correct, total, NULL, NULL);
		const double top1Pct = (100.0 * static_cast<double>(correct)) / static_cast<double>(total);
		adamTop1Sum += top1Pct;
		adamMacroSum += static_cast<double>(evalCb.last.totalAccuracy);
		adamLossSum += static_cast<double>(evalCb.last.totalError);

		printf("  [AdamW 1->4->1] seed=%llu converged_ep=%u loss=%.6f top1=%u/%u (%.1f%%) preds=[%.4f, %.4f, %.4f, %.4f]\n",
		       static_cast<unsigned long long>(seeds[s]),
		       cb.firstPerfectEpoch,
		       evalCb.last.totalError,
		       correct, total, top1Pct,
		       preds[0][0], preds[1][0], preds[2][0], preds[3][0]);

		ASSERT("XOR AdamW achieves 100% top-1 accuracy", correct == 4u);
		ASSERT("XOR AdamW achieves 100% macro accuracy", evalCb.last.totalAccuracy >= 99.9f);
		ASSERT("XOR AdamW achieves BCE loss < 0.02", evalCb.last.totalError < 0.02f);
		ASSERT("XOR (0,0) -> p < 0.1", preds[0][0] < 0.1f);
		ASSERT("XOR (0,1) -> p > 0.9", preds[1][0] > 0.9f);
		ASSERT("XOR (1,0) -> p > 0.9", preds[2][0] > 0.9f);
		ASSERT("XOR (1,1) -> p < 0.1", preds[3][0] < 0.1f);
	}
	{
		BenchmarkSummaryRow row;
		row.label = "XOR (AdamW)";
		row.arch = "1->4->1 (52 real)";
		row.seeds = 5u;
		row.epochs = 600u;
		row.meanTrainTop1Pct = adamTop1Sum / 5.0;
		row.meanTestTop1Pct = adamTop1Sum / 5.0;
		row.meanTestMacroAccPct = adamMacroSum / 5.0;
		row.meanTestLoss = adamLossSum / 5.0;
		summary.push_back(row);
	}

	// 1b. SGD + momentum verification on XOR (1->4->1)
	{
		glades::NNInfo* info = make_qnn_info(
		    "qnn_xor_sgd", /*batchSize*/ 4, /*lr*/ 0.2f, /*momentum*/ 0.9f, /*wd2*/ 0.0f,
		    hidden4, glades::GMath::TANH, /*outSize*/ 1, glades::OutputLayerInfo::CLASSIFICATION, glades::GMath::SIGMOID);

		glades::NNetwork net(info, glades::NNetwork::TYPE_QNN);
		delete info;

		net.setSeed(101ULL);
		glades::Terminator term;
		term.setEpoch(800);
		ASSERT("XOR SGD setTerminator succeeds", net.setTerminator(term).ok());

		CaptureLastEpochCb trainCb;
		glades::NNetworkStatus st = net.train(&xorDi, &trainCb);
		ASSERT("XOR SGD+momentum train succeeds", st.ok());

		CaptureLastEpochCb evalCb;
		st = net.test(&xorDi, &evalCb);
		ASSERT("XOR SGD+momentum test succeeds", st.ok());

		std::vector< std::vector<float> > preds;
		unsigned int correct = 0u, total = 0u;
		eval_split_predictions(net, xorDi, false, preds, correct, total, NULL, NULL);
		const double top1Pct = (100.0 * static_cast<double>(correct)) / static_cast<double>(total);
		printf("  [SGD+M 1->4->1] seed=101 converged_ep=%u loss=%.6f top1=%u/%u (%.1f%%) preds=[%.4f, %.4f, %.4f, %.4f]\n",
		       trainCb.firstPerfectEpoch,
		       evalCb.last.totalError, correct, total, top1Pct,
		       preds[0][0], preds[1][0], preds[2][0], preds[3][0]);

		ASSERT("XOR SGD+momentum achieves 100% top-1 accuracy", correct == 4u);
		ASSERT("XOR SGD+momentum achieves BCE loss < 0.02", evalCb.last.totalError < 0.02f);

		BenchmarkSummaryRow row;
		row.label = "XOR (SGD+M)";
		row.arch = "1->4->1 (52 real)";
		row.seeds = 1u;
		row.epochs = 800u;
		row.meanTrainTop1Pct = top1Pct;
		row.meanTestTop1Pct = top1Pct;
		row.meanTestMacroAccPct = static_cast<double>(evalCb.last.totalAccuracy);
		row.meanTestLoss = static_cast<double>(evalCb.last.totalError);
		summary.push_back(row);
	}

	// 1c. Minimal 1 -> 1 -> 1 quaternion architecture (single hidden quaternion!)
	{
		std::vector<unsigned int> hidden1;
		hidden1.push_back(1u);
		glades::NNInfo* info = make_qnn_info(
		    "qnn_xor_minimal_1_1_1", /*batchSize*/ 4, /*lr*/ 0.05f, /*momentum*/ 0.0f, /*wd2*/ 0.0f,
		    hidden1, glades::GMath::TANH, /*outSize*/ 1, glades::OutputLayerInfo::CLASSIFICATION, glades::GMath::SIGMOID);

		glades::NNetwork net(info, glades::NNetwork::TYPE_QNN);
		delete info;

		net.setSeed(2026ULL);
		glades::TrainingConfig cfg = net.getTrainingConfig();
		cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
		ASSERT("XOR 1->1->1 setTrainingConfig succeeds", net.setTrainingConfig(cfg).ok());

		glades::Terminator term;
		term.setEpoch(1500);
		ASSERT("XOR 1->1->1 setTerminator succeeds", net.setTerminator(term).ok());

		CaptureLastEpochCb trainCb;
		glades::NNetworkStatus st = net.train(&xorDi, &trainCb);
		ASSERT("XOR 1->1->1 train succeeds", st.ok());

		CaptureLastEpochCb evalCb;
		st = net.test(&xorDi, &evalCb);
		ASSERT("XOR 1->1->1 test succeeds", st.ok());

		std::vector< std::vector<float> > preds;
		unsigned int correct = 0u, total = 0u;
		eval_split_predictions(net, xorDi, false, preds, correct, total, NULL, NULL);
		const double top1Pct = (100.0 * static_cast<double>(correct)) / static_cast<double>(total);
		printf("  [Minimal 1->1->1] seed=2026 converged_ep=%u loss=%.6f top1=%u/%u (%.1f%%) preds=[%.4f, %.4f, %.4f, %.4f]\n",
		       trainCb.firstPerfectEpoch,
		       evalCb.last.totalError, correct, total, top1Pct,
		       preds[0][0], preds[1][0], preds[2][0], preds[3][0]);

		ASSERT("Minimal 1->1->1 QNN solves XOR with 100% top-1 accuracy", correct == 4u);
		ASSERT("Minimal 1->1->1 QNN achieves BCE loss < 0.02", evalCb.last.totalError < 0.02f);

		BenchmarkSummaryRow row;
		row.label = "XOR (Minimal)";
		row.arch = "1->1->1 (16 real)";
		row.seeds = 1u;
		row.epochs = 1500u;
		row.meanTrainTop1Pct = top1Pct;
		row.meanTestTop1Pct = top1Pct;
		row.meanTestMacroAccPct = static_cast<double>(evalCb.last.totalAccuracy);
		row.meanTestLoss = static_cast<double>(evalCb.last.totalError);
		summary.push_back(row);
	}

	printf("[PASS] XOR gate convergence verified.\n");
}

static void test_qnn_iris(std::vector<BenchmarkSummaryRow>& summary)
{
	printf("-----------------------------------\n");
	printf("QNN Phase 2: Iris Classification & Persistence\n");
	printf("-----------------------------------\n");

	glades::NumberInput::TrainTestSplitConfig splitCfg;
	splitCfg.testFraction = 0.2f;
	splitCfg.shuffle = true;
	splitCfg.stratify = true;
	splitCfg.seed = 42ULL;

	glades::QuaternionInput irisRealDi;
	ASSERT("Load Iris via QuaternionInput::loadIris (PACK4 + REAL_PART) succeeds",
	       irisRealDi.loadIris("datasets/iris.data",
	                           splitCfg,
	                           glades::GMath::ZSCORE,
	                           glades::QUAT_PACK_GROUPS_OF_4,
	                           glades::QUAT_READOUT_REAL_PART));
	ASSERT("Iris train size is 120", irisRealDi.getTrainSize() == 120u);
	ASSERT("Iris test size is 30", irisRealDi.getTestSize() == 30u);
	ASSERT("Iris feature count is 4", irisRealDi.getFeatureCount() == 4u);
	ASSERT("Iris quat input count is 1", irisRealDi.getQuatInputCount() == 1u);
	ASSERT("Iris quat output count is 3", irisRealDi.getQuatOutputCount() == 3u);

	// 2a. Primary architecture: Input(4) -> 1 quat -> Hidden(6 quats, TANH) -> Output(3 quats, Softmax CE)
	std::vector<unsigned int> hidden6;
	hidden6.push_back(6u);

	const uint64_t seeds[5] = { 42ULL, 1337ULL, 2024ULL, 7777ULL, 9001ULL };
	double primaryTrainTop1Sum = 0.0;
	double primaryTestTop1Sum = 0.0;
	double primaryTestMacroSum = 0.0;
	double primaryTestLossSum = 0.0;

	for (unsigned int s = 0u; s < 5u; ++s)
	{
		glades::NNInfo* info = make_qnn_info(
		    "qnn_iris_primary", /*batchSize*/ 16, /*lr*/ 0.02f, /*momentum*/ 0.0f, /*wd2*/ 1e-4f,
		    hidden6, glades::GMath::TANH, /*outSize*/ 3, glades::OutputLayerInfo::CLASSIFICATION, glades::GMath::LINEAR);

		glades::NNetwork net(info, glades::NNetwork::TYPE_QNN);
		delete info;

		net.setSeed(seeds[s]);
		glades::TrainingConfig cfg = net.getTrainingConfig();
		cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
		cfg.qnn.inputPacking = glades::QuaternionConfig::PACK_GROUPS_OF_4;
		cfg.qnn.outputReadout = glades::QuaternionConfig::READOUT_REAL_PART;
		ASSERT("Iris setTrainingConfig succeeds", net.setTrainingConfig(cfg).ok());

		glades::Terminator term;
		term.setEpoch(300);
		ASSERT("Iris setTerminator succeeds", net.setTerminator(term).ok());

		CaptureLastEpochCb trainCb;
		glades::NNetworkStatus st = net.train(&irisRealDi, &trainCb);
		ASSERT("Iris primary train succeeds", st.ok());

		CaptureLastEpochCb testCb;
		st = net.test(&irisRealDi, &testCb);
		ASSERT("Iris primary test succeeds", st.ok());

		std::vector< std::vector<float> > trainPreds;
		unsigned int trainCorrect = 0u, trainTotal = 0u;
		eval_split_predictions(net, irisRealDi, true, trainPreds, trainCorrect, trainTotal, NULL, NULL);
		const double trainTop1Pct = (100.0 * static_cast<double>(trainCorrect)) / static_cast<double>(trainTotal);

		std::vector< std::vector<float> > testPreds;
		unsigned int testCorrect = 0u, testTotal = 0u;
		std::vector<unsigned int> perClsCorrect, perClsTotal;
		eval_split_predictions(net, irisRealDi, false, testPreds, testCorrect, testTotal, &perClsCorrect, &perClsTotal);
		const double testTop1Pct = (100.0 * static_cast<double>(testCorrect)) / static_cast<double>(testTotal);

		printf("  [Primary 1->6->3] seed=%llu ep1_loss=%.4f->%.4f train_top1=%u/%u (%.2f%%) test_top1=%u/%u (%.2f%%) test_macro=%.2f%% test_loss=%.4f\n",
		       static_cast<unsigned long long>(seeds[s]),
		       trainCb.first.totalError, trainCb.last.totalError,
		       trainCorrect, trainTotal, trainTop1Pct,
		       testCorrect, testTotal, testTop1Pct,
		       testCb.last.totalAccuracy, testCb.last.totalError);

		primaryTrainTop1Sum += trainTop1Pct;
		primaryTestTop1Sum += testTop1Pct;
		primaryTestMacroSum += static_cast<double>(testCb.last.totalAccuracy);
		primaryTestLossSum += static_cast<double>(testCb.last.totalError);

		if (s == 0u)
		{
			const std::vector<std::string>& labels = irisRealDi.getClassLabels();
			printf("    Per-class held-out (seed=42):");
			for (unsigned int c = 0u; c < 3u; ++c)
			{
				const char* cname = (c < labels.size()) ? labels[c].c_str() : "class";
				printf(" %s=%u/%u", cname, perClsCorrect[c], perClsTotal[c]);
			}
			printf("\n");

			ASSERT("Iris canonical seed train top-1 >= 96%", trainTop1Pct >= 96.0);
			ASSERT("Iris canonical seed test top-1 >= 93.3% (>= 28/30)", testCorrect >= 28u);

			// Save/load round-trip on the canonical trained Iris QNN model.
			const std::string modelName = "ut_qnn_iris_roundtrip";
			remove_model_dir_best_effort(modelName);
			st = net.saveModel(modelName, &irisRealDi);
			ASSERT("Iris QNN saveModel succeeds", st.ok());

			glades::NNetwork loaded;
			st = loaded.loadModel(modelName, &irisRealDi);
			ASSERT("Iris QNN loadModel succeeds", st.ok());
			ASSERT("Loaded netType is TYPE_QNN", loaded.getNetType() == glades::NNetwork::TYPE_QNN);

			CaptureLastEpochCb loadedCb;
			st = loaded.test(&irisRealDi, &loadedCb);
			ASSERT("Loaded QNN test succeeds", st.ok());
			ASSERT("Loaded QNN test accuracy matches original",
			       near_f(loadedCb.last.totalAccuracy, testCb.last.totalAccuracy, 1e-5f));

			std::vector< std::vector<float> > loadedPreds;
			eval_single_row_predictions(loaded, irisRealDi, loadedPreds);
			float maxDiff = 0.0f;
			for (size_t r = 0u; r < testPreds.size(); ++r)
			{
				for (size_t k = 0u; k < testPreds[r].size(); ++k)
				{
					const float d = fabsf(testPreds[r][k] - loadedPreds[r][k]);
					if (d > maxDiff)
						maxDiff = d;
				}
			}
			printf("  [Save/Load round-trip] max_pred_diff=%.8g\n", maxDiff);
			ASSERT("Save/load round-trip predictions are bit-exact", maxDiff == 0.0f);

			// Also verify loadModelDirectory (strict v3 deployment read).
			const std::string absDir = current_working_dir() + "/database/models/" + modelName;
			glades::NNetwork deployed;
			st = deployed.loadModelDirectory(absDir, &irisRealDi);
			ASSERT("Iris QNN loadModelDirectory succeeds", st.ok());
			CaptureLastEpochCb deployedCb;
			st = deployed.test(&irisRealDi, &deployedCb);
			ASSERT("Deployed QNN test succeeds", st.ok());
			ASSERT("Deployed QNN test accuracy matches original",
			       near_f(deployedCb.last.totalAccuracy, testCb.last.totalAccuracy, 1e-5f));

			remove_model_dir_best_effort(modelName);
		}
	}

	const double meanPrimaryTop1 = primaryTestTop1Sum / 5.0;
	printf("  [Primary 1->6->3] 5-seed mean held-out top-1 = %.2f%% (macro = %.2f%%)\n",
	       meanPrimaryTop1, primaryTestMacroSum / 5.0);
	ASSERT("Iris 5-seed mean held-out top-1 accuracy >= 93%", meanPrimaryTop1 >= 93.0);

	{
		BenchmarkSummaryRow row;
		row.label = "Iris (Real-Part)";
		row.arch = "1->6->3 (123 active)";
		row.seeds = 5u;
		row.epochs = 300u;
		row.meanTrainTop1Pct = primaryTrainTop1Sum / 5.0;
		row.meanTestTop1Pct = meanPrimaryTop1;
		row.meanTestMacroAccPct = primaryTestMacroSum / 5.0;
		row.meanTestLoss = primaryTestLossSum / 5.0;
		summary.push_back(row);
	}

	// 2b. Compact split-readout variant across 5 seeds:
	//     Input(4)->1 quat -> Hidden(6 quats) -> Output(1 quat, 3 class lanes r,i,j)
	glades::QuaternionInput irisSplitDi;
	ASSERT("Load Iris via QuaternionInput::loadIris (PACK4 + SPLIT_COMPONENTS) succeeds",
	       irisSplitDi.loadIris("datasets/iris.data",
	                            splitCfg,
	                            glades::GMath::ZSCORE,
	                            glades::QUAT_PACK_GROUPS_OF_4,
	                            glades::QUAT_READOUT_SPLIT_COMPONENTS));
	ASSERT("Iris compact quat output count is 1", irisSplitDi.getQuatOutputCount() == 1u);

	double compactTrainTop1Sum = 0.0;
	double compactTestTop1Sum = 0.0;
	double compactTestMacroSum = 0.0;
	double compactTestLossSum = 0.0;

	for (unsigned int s = 0u; s < 5u; ++s)
	{
		glades::NNInfo* info = make_qnn_info(
		    "qnn_iris_compact_split", /*batchSize*/ 16, /*lr*/ 0.02f, /*momentum*/ 0.0f, /*wd2*/ 1e-4f,
		    hidden6, glades::GMath::TANH, /*outSize*/ 3, glades::OutputLayerInfo::CLASSIFICATION, glades::GMath::LINEAR);

		glades::NNetwork net(info, glades::NNetwork::TYPE_QNN);
		delete info;

		net.setSeed(seeds[s]);
		glades::TrainingConfig cfg = net.getTrainingConfig();
		cfg.optimizer.type = glades::OptimizerConfig::ADAMW;
		cfg.qnn.inputPacking = glades::QuaternionConfig::PACK_GROUPS_OF_4;
		cfg.qnn.outputReadout = glades::QuaternionConfig::READOUT_SPLIT_COMPONENTS;
		ASSERT("Iris compact setTrainingConfig succeeds", net.setTrainingConfig(cfg).ok());

		glades::Terminator term;
		term.setEpoch(300);
		ASSERT("Iris compact setTerminator succeeds", net.setTerminator(term).ok());

		CaptureLastEpochCb trainCb;
		ASSERT("Iris compact train succeeds", net.train(&irisSplitDi, &trainCb).ok());
		CaptureLastEpochCb testCb;
		ASSERT("Iris compact test succeeds", net.test(&irisSplitDi, &testCb).ok());

		std::vector< std::vector<float> > trainPreds, testPreds;
		unsigned int trainCorrect = 0u, trainTotal = 0u;
		unsigned int testCorrect = 0u, testTotal = 0u;
		eval_split_predictions(net, irisSplitDi, true, trainPreds, trainCorrect, trainTotal, NULL, NULL);
		eval_split_predictions(net, irisSplitDi, false, testPreds, testCorrect, testTotal, NULL, NULL);
		const double trainTop1Pct = (100.0 * static_cast<double>(trainCorrect)) / static_cast<double>(trainTotal);
		const double testTop1Pct = (100.0 * static_cast<double>(testCorrect)) / static_cast<double>(testTotal);

		printf("  [Compact 1->6->1] seed=%llu train_top1=%u/%u (%.2f%%) test_top1=%u/%u (%.2f%%) test_macro=%.2f%% test_loss=%.4f\n",
		       static_cast<unsigned long long>(seeds[s]),
		       trainCorrect, trainTotal, trainTop1Pct,
		       testCorrect, testTotal, testTop1Pct,
		       testCb.last.totalAccuracy, testCb.last.totalError);

		compactTrainTop1Sum += trainTop1Pct;
		compactTestTop1Sum += testTop1Pct;
		compactTestMacroSum += static_cast<double>(testCb.last.totalAccuracy);
		compactTestLossSum += static_cast<double>(testCb.last.totalError);

		if (s == 0u)
		{
			ASSERT("Iris compact split-readout canonical seed test top-1 >= 93.3% (>= 28/30)", testCorrect >= 28u);

			// Verify split-readout boundary mode round-trips through saveModel/loadModel.
			const std::string compactName = "ut_qnn_iris_compact_roundtrip";
			remove_model_dir_best_effort(compactName);
			ASSERT("Iris compact saveModel succeeds", net.saveModel(compactName, &irisSplitDi).ok());

			glades::NNetwork loadedCompact;
			ASSERT("Iris compact loadModel succeeds", loadedCompact.loadModel(compactName, &irisSplitDi).ok());
			ASSERT("Loaded compact preserves READOUT_SPLIT_COMPONENTS",
			       loadedCompact.getTrainingConfig().qnn.outputReadout == glades::QuaternionConfig::READOUT_SPLIT_COMPONENTS);
			CaptureLastEpochCb loadedCompactCb;
			ASSERT("Loaded compact test succeeds", loadedCompact.test(&irisSplitDi, &loadedCompactCb).ok());
			ASSERT("Loaded compact test acc matches", near_f(loadedCompactCb.last.totalAccuracy, testCb.last.totalAccuracy, 1e-5f));
			remove_model_dir_best_effort(compactName);
		}
	}

	const double meanCompactTop1 = compactTestTop1Sum / 5.0;
	printf("  [Compact 1->6->1] 5-seed mean held-out top-1 = %.2f%% (macro = %.2f%%)\n",
	       meanCompactTop1, compactTestMacroSum / 5.0);
	ASSERT("Iris compact 5-seed mean held-out top-1 accuracy >= 93%", meanCompactTop1 >= 93.0);

	{
		BenchmarkSummaryRow row;
		row.label = "Iris (Split-Lane)";
		row.arch = "1->6->1 (76 real)";
		row.seeds = 5u;
		row.epochs = 300u;
		row.meanTrainTop1Pct = compactTrainTop1Sum / 5.0;
		row.meanTestTop1Pct = meanCompactTop1;
		row.meanTestMacroAccPct = compactTestMacroSum / 5.0;
		row.meanTestLoss = compactTestLossSum / 5.0;
		summary.push_back(row);
	}

	printf("[PASS] Iris classification and persistence verified.\n");
}

struct QuatD
{
	double r, i, j, k;
	QuatD() : r(0.0), i(0.0), j(0.0), k(0.0) {}
	QuatD(double r_, double i_, double j_, double k_) : r(r_), i(i_), j(j_), k(k_) {}
	explicit QuatD(const glades::Quaternion& q) : r(q.r), i(q.i), j(q.j), k(q.k) {}

	double component(unsigned int lane) const
	{
		switch (lane)
		{
		case 0u: return r;
		case 1u: return i;
		case 2u: return j;
		case 3u: return k;
		default: return 0.0;
		}
	}
	void setComponent(unsigned int lane, double v)
	{
		switch (lane)
		{
		case 0u: r = v; break;
		case 1u: i = v; break;
		case 2u: j = v; break;
		case 3u: k = v; break;
		default: break;
		}
	}
	static QuatD mul(const QuatD& a, const QuatD& b)
	{
		return QuatD(
		    (a.r * b.r) - (a.i * b.i) - (a.j * b.j) - (a.k * b.k),
		    (a.r * b.i) + (a.i * b.r) + (a.j * b.k) - (a.k * b.j),
		    (a.r * b.j) - (a.i * b.k) + (a.j * b.r) + (a.k * b.i),
		    (a.r * b.k) + (a.i * b.j) - (a.j * b.i) + (a.k * b.r));
	}
};

double squash_double(double x, int actFx)
{
	if (actFx == glades::GMath::TANH || actFx == glades::GMath::TANHP)
		return tanh(x);
	if (actFx == glades::GMath::SIGMOID || actFx == glades::GMath::SIGMOIDP)
		return 1.0 / (1.0 + exp(-x));
	return x; // LINEAR
}

double eval_qnn_loss_double(const std::vector<double>& xIn,
                            const std::vector<double>& yExp,
                            const std::vector<unsigned int>& qSizes,
                            const std::vector< std::vector<QuatD> >& Ws,
                            const std::vector< std::vector<QuatD> >& Bs,
                            int hiddenAct,
                            int outAct,
                            int outType,
                            glades::QuaternionConfig::InputPackingMode packing,
                            glades::QuaternionConfig::OutputReadoutMode readout)
{
	const unsigned int qIn = qSizes[0];
	const unsigned int numLayers = static_cast<unsigned int>(qSizes.size());
	const unsigned int lastLayer = numLayers - 1u;
	const unsigned int realOut = static_cast<unsigned int>(yExp.size());
	const bool useSoftmax = (outType == glades::OutputLayerInfo::CLASSIFICATION) && (realOut > 1u);

	std::vector< std::vector<QuatD> > a(numLayers);
	std::vector< std::vector<QuatD> > z(numLayers);
	for (unsigned int l = 0u; l < numLayers; ++l)
	{
		a[l].resize(qSizes[l]);
		z[l].resize(qSizes[l]);
	}

	if (packing == glades::QuaternionConfig::PACK_GROUPS_OF_4)
	{
		for (size_t f = 0u; f < xIn.size(); ++f)
			a[0][f / 4u].setComponent(static_cast<unsigned int>(f % 4u), xIn[f]);
	}
	else
	{
		for (size_t f = 0u; f < xIn.size() && f < qIn; ++f)
			a[0][f] = QuatD(xIn[f], 0.0, 0.0, 0.0);
	}

	for (unsigned int t = 0u; t + 1u < numLayers; ++t)
	{
		const unsigned int qCur = qSizes[t];
		const unsigned int qNext = qSizes[t + 1u];
		const bool isLast = (t + 1u == lastLayer);
		const int act = isLast ? outAct : hiddenAct;
		for (unsigned int m = 0u; m < qNext; ++m)
		{
			QuatD acc = Bs[t][m];
			const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(qCur);
			for (unsigned int n = 0u; n < qCur; ++n)
			{
				const QuatD prod = QuatD::mul(Ws[t][rowOff + n], a[t][n]);
				acc.r += prod.r;
				acc.i += prod.i;
				acc.j += prod.j;
				acc.k += prod.k;
			}
			z[t + 1u][m] = acc;
			if (useSoftmax && isLast)
			{
				a[t + 1u][m] = acc;
			}
			else
			{
				a[t + 1u][m] = QuatD(
				    squash_double(acc.r, act),
				    squash_double(acc.i, act),
				    squash_double(acc.j, act),
				    squash_double(acc.k, act));
			}
		}
	}

	std::vector<double> out(realOut, 0.0);
	if (useSoftmax)
	{
		double maxLogit = -1e300;
		for (unsigned int k = 0u; k < realOut; ++k)
		{
			const double v = (readout == glades::QuaternionConfig::READOUT_SPLIT_COMPONENTS)
			    ? z[lastLayer][k / 4u].component(k % 4u)
			    : z[lastLayer][k].r;
			out[k] = v;
			if (v > maxLogit)
				maxLogit = v;
		}
		double sumExp = 0.0;
		for (unsigned int k = 0u; k < realOut; ++k)
		{
			out[k] = exp(out[k] - maxLogit);
			sumExp += out[k];
		}
		double loss = 0.0;
		for (unsigned int k = 0u; k < realOut; ++k)
		{
			const double p = out[k] / sumExp;
			loss += -yExp[k] * log(p);
		}
		return loss;
	}

	for (unsigned int k = 0u; k < realOut; ++k)
	{
		out[k] = (readout == glades::QuaternionConfig::READOUT_SPLIT_COMPONENTS)
		    ? a[lastLayer][k / 4u].component(k % 4u)
		    : a[lastLayer][k].r;
	}

	if (outType == glades::OutputLayerInfo::REGRESSION)
	{
		// Matches GMath::costErrDer(y, p, REGRESSION) = 2 * (p - y), i.e., sum_k (p_k - y_k)^2.
		double sse = 0.0;
		for (unsigned int k = 0u; k < realOut; ++k)
		{
			const double d = out[k] - yExp[k];
			sse += d * d;
		}
		return sse;
	}

	// Binary cross-entropy for single-output sigmoid classification.
	const double p = out[0];
	const double y = yExp[0];
	return -(y * log(p) + (1.0 - y) * log(1.0 - p));
}

} // namespace

void QNNUnitTest()
{
	glades::init();

	printf("============================================================\n");
	printf("Quaternion Neural Network (TYPE_QNN) Unit Test Suite\n");
	printf("============================================================\n");

	// Phase 0a: Quaternion algebra identities
	test_quaternion_algebra();

	// Phase 0b: Central finite-difference gradient check across all (r,i,j,k) of W, b, a
	printf("-----------------------------------\n");
	printf("QNN Phase 0b: Central Finite-Difference Gradient Check\n");
	printf("-----------------------------------\n");
	{
		struct GradCheckCase
		{
			const char* label;
			unsigned int realIn;
			unsigned int realOut;
			int outType;
			int outAct;
			glades::QuaternionConfig::InputPackingMode packing;
			glades::QuaternionConfig::OutputReadoutMode readout;
			float lossScaleForFd; // matches costErrDer scaling vs Trainer::overallTotalError
		};

		const GradCheckCase cases[4] = {
			{ "REGRESSION PACK4 READOUT_REAL (D_in=5, D_out=3)",
			  5u, 3u, glades::OutputLayerInfo::REGRESSION, glades::GMath::LINEAR,
			  glades::QuaternionConfig::PACK_GROUPS_OF_4, glades::QuaternionConfig::READOUT_REAL_PART, 3.0f },
			{ "CLASSIFICATION Softmax PACK4 READOUT_REAL (D_in=4, D_out=3)",
			  4u, 3u, glades::OutputLayerInfo::CLASSIFICATION, glades::GMath::LINEAR,
			  glades::QuaternionConfig::PACK_GROUPS_OF_4, glades::QuaternionConfig::READOUT_REAL_PART, 1.0f },
			{ "CLASSIFICATION Softmax PACK4 READOUT_SPLIT (D_in=4, D_out=3)",
			  4u, 3u, glades::OutputLayerInfo::CLASSIFICATION, glades::GMath::LINEAR,
			  glades::QuaternionConfig::PACK_GROUPS_OF_4, glades::QuaternionConfig::READOUT_SPLIT_COMPONENTS, 1.0f },
			{ "CLASSIFICATION Binary BCE ONE_PER_FEAT READOUT_REAL (D_in=2, D_out=1)",
			  2u, 1u, glades::OutputLayerInfo::CLASSIFICATION, glades::GMath::SIGMOID,
			  glades::QuaternionConfig::ONE_QUAT_PER_FEATURE, glades::QuaternionConfig::READOUT_REAL_PART, 1.0f }
		};

		const double epsD = 1e-4;
		const float epsF = 1e-3f; // optimal float32 central-FD step (~ u^(1/3))
		const float probeLr = 1.0f;

		for (unsigned int ci = 0u; ci < 4u; ++ci)
		{
			const GradCheckCase& tc = cases[ci];

			glades::NumberInput di;
			di.trainMatrix = shmea::GMatrix(1, shmea::GVector<float>(tc.realIn, 0.0f));
			di.trainExpectedMatrix = shmea::GMatrix(1, shmea::GVector<float>(tc.realOut, 0.0f));
			std::vector<double> xD(tc.realIn, 0.0);
			std::vector<double> yD(tc.realOut, 0.0);
			for (unsigned int f = 0u; f < tc.realIn; ++f)
			{
				const float xv = -0.6f + 0.37f * static_cast<float>(f);
				di.trainMatrix[0][f] = xv;
				xD[f] = static_cast<double>(xv);
			}
			if (tc.outType == glades::OutputLayerInfo::REGRESSION)
			{
				for (unsigned int k = 0u; k < tc.realOut; ++k)
				{
					const float yv = -0.4f + 0.55f * static_cast<float>(k);
					di.trainExpectedMatrix[0][k] = yv;
					yD[k] = static_cast<double>(yv);
				}
			}
			else if (tc.realOut == 1u)
			{
				di.trainExpectedMatrix[0][0] = 1.0f;
				yD[0] = 1.0;
			}
			else
			{
				di.trainExpectedMatrix[0][1] = 1.0f; // one-hot class 1
				yD[1] = 1.0;
			}
			di.testMatrix = di.trainMatrix;
			di.testExpectedMatrix = di.trainExpectedMatrix;
			di.loaded = true;

			std::vector<unsigned int> hidden2;
			hidden2.push_back(3u);
			hidden2.push_back(2u);
			glades::NNInfo* info = make_qnn_info(
			    "qnn_gradcheck", /*batchSize*/ 1, /*lr*/ probeLr, /*momentum*/ 0.0f, /*wd2*/ 0.0f,
			    hidden2, glades::GMath::TANH, static_cast<int>(tc.realOut), tc.outType, tc.outAct);

			glades::NNetwork net(info, glades::NNetwork::TYPE_QNN);
			delete info;
			net.setSeed(700ULL + ci);

			glades::TrainingConfig cfg = net.getTrainingConfig();
			cfg.optimizer.type = glades::OptimizerConfig::SGD_MOMENTUM;
			cfg.perElementGradClip = 0.0f;
			cfg.globalGradClipNorm = 0.0f;
			cfg.qnn.inputPacking = tc.packing;
			cfg.qnn.outputReadout = tc.readout;
			ASSERT("Grad-check setTrainingConfig succeeds", net.setTrainingConfig(cfg).ok());

			// Initialize weights via one read-only test pass.
			CaptureLastEpochCb initCb;
			ASSERT("Grad-check init test() succeeds", net.test(&di, &initCb).ok());
			ASSERT("Grad-check tensorQnn initialized", net.tensorQnn.initialized);

			// Snapshot initial weights & biases in both float32 and double.
			const size_t numT = net.tensorQnn.T.size();
			std::vector< std::vector<glades::Quaternion> > origW(numT);
			std::vector< std::vector<glades::Quaternion> > origB(numT);
			std::vector< std::vector<QuatD> > wD(numT);
			std::vector< std::vector<QuatD> > bD(numT);
			for (size_t t = 0u; t < numT; ++t)
			{
				origW[t] = net.tensorQnn.T[t].W;
				origB[t] = net.tensorQnn.T[t].bias;
				wD[t].resize(origW[t].size());
				for (size_t idx = 0u; idx < origW[t].size(); ++idx)
					wD[t][idx] = QuatD(origW[t][idx]);
				bD[t].resize(origB[t].size());
				for (size_t m = 0u; m < origB[t].size(); ++m)
					bD[t][m] = QuatD(origB[t][m]);
			}

			// Verify NNetwork::test() forward loss matches the double-precision reference to float32 precision.
			const double refLoss = eval_qnn_loss_double(
			    xD, yD, net.tensorQnn.qSizes, wD, bD,
			    glades::GMath::TANH, tc.outAct, tc.outType, tc.packing, tc.readout);
			const double netLoss = static_cast<double>(initCb.last.totalError) * static_cast<double>(tc.lossScaleForFd);
			ASSERT("Forward loss matches double reference (< 1e-5)", fabs(netLoss - refLoss) < 1e-5);

			// Run 1 epoch of plain SGD (mf=0, wd=0, clip=0, lr=1.0) so velocity buffers
			// hold the exact analytic gradients computed by SGDHelper_QNN without subtraction noise.
			glades::Terminator term;
			term.setEpoch(1);
			ASSERT("Grad-check setTerminator succeeds", net.setTerminator(term).ok());
			ASSERT("Grad-check 1-step train succeeds", net.train(&di).ok());

			std::vector< std::vector<glades::Quaternion> > analyticGW(numT);
			std::vector< std::vector<glades::Quaternion> > analyticGB(numT);
			for (size_t t = 0u; t < numT; ++t)
			{
				analyticGW[t].resize(origW[t].size());
				for (size_t idx = 0u; idx < origW[t].size(); ++idx)
					analyticGW[t][idx] = net.tensorQnn.T[t].vW[idx] / probeLr;
				analyticGB[t].resize(origB[t].size());
				for (size_t m = 0u; m < origB[t].size(); ++m)
					analyticGB[t][m] = net.tensorQnn.T[t].vBias[m] / probeLr;
			}
			const std::vector<glades::Quaternion> analyticGIn = net.tensorQnn.delta[0];

			// Restore exact original weights, biases, and zero velocities.
			for (size_t t = 0u; t < numT; ++t)
			{
				net.tensorQnn.T[t].W = origW[t];
				net.tensorQnn.T[t].bias = origB[t];
				std::fill(net.tensorQnn.T[t].vW.begin(), net.tensorQnn.T[t].vW.end(), glades::Quaternion::zero());
				std::fill(net.tensorQnn.T[t].vBias.begin(), net.tensorQnn.T[t].vBias.end(), glades::Quaternion::zero());
			}

			double maxRelErrD = 0.0;
			double maxAbsErrF = 0.0;

			// 1) Central finite-difference at eps=1e-4 (double-precision loss evaluation) across every (r,i,j,k) of W, b, a^(0).
			// 2) End-to-end float32 NNetwork::test() central FD at eps=1e-3 across every (r,i,j,k) of W, b, a^(0).
			for (size_t t = 0u; t < numT; ++t)
			{
				for (size_t idx = 0u; idx < net.tensorQnn.T[t].W.size(); ++idx)
				{
					for (unsigned int lane = 0u; lane < 4u; ++lane)
					{
						const double baseD = wD[t][idx].component(lane);
						wD[t][idx].setComponent(lane, baseD + epsD);
						const double lPD = eval_qnn_loss_double(xD, yD, net.tensorQnn.qSizes, wD, bD, glades::GMath::TANH, tc.outAct, tc.outType, tc.packing, tc.readout);
						wD[t][idx].setComponent(lane, baseD - epsD);
						const double lMD = eval_qnn_loss_double(xD, yD, net.tensorQnn.qSizes, wD, bD, glades::GMath::TANH, tc.outAct, tc.outType, tc.packing, tc.readout);
						wD[t][idx].setComponent(lane, baseD);

						const double fdD = (lPD - lMD) / (2.0 * epsD);
						const double ag = static_cast<double>(analyticGW[t][idx].component(lane));
						const double relErrD = fabs(fdD - ag) / std::max(1e-5, std::max(fabs(fdD), fabs(ag)));
						if (relErrD > maxRelErrD)
							maxRelErrD = relErrD;

						const float baseF = net.tensorQnn.T[t].W[idx].component(lane);
						const float pVal = baseF + epsF;
						const float mVal = baseF - epsF;
						net.tensorQnn.T[t].W[idx].setComponent(lane, pVal);
						CaptureLastEpochCb cbP;
						ASSERT("FD W+ test succeeds", net.test(&di, &cbP).ok());
						const double lPF = static_cast<double>(cbP.last.totalError) * static_cast<double>(tc.lossScaleForFd);
						net.tensorQnn.T[t].W[idx].setComponent(lane, mVal);
						CaptureLastEpochCb cbM;
						ASSERT("FD W- test succeeds", net.test(&di, &cbM).ok());
						const double lMF = static_cast<double>(cbM.last.totalError) * static_cast<double>(tc.lossScaleForFd);
						net.tensorQnn.T[t].W[idx].setComponent(lane, baseF);
						const double fdF = (lPF - lMF) / (static_cast<double>(pVal) - static_cast<double>(mVal));
						const double absErrF = fabs(fdF - ag);
						if (absErrF > maxAbsErrF)
							maxAbsErrF = absErrF;
					}
				}

				for (size_t m = 0u; m < net.tensorQnn.T[t].bias.size(); ++m)
				{
					for (unsigned int lane = 0u; lane < 4u; ++lane)
					{
						const double baseD = bD[t][m].component(lane);
						bD[t][m].setComponent(lane, baseD + epsD);
						const double lPD = eval_qnn_loss_double(xD, yD, net.tensorQnn.qSizes, wD, bD, glades::GMath::TANH, tc.outAct, tc.outType, tc.packing, tc.readout);
						bD[t][m].setComponent(lane, baseD - epsD);
						const double lMD = eval_qnn_loss_double(xD, yD, net.tensorQnn.qSizes, wD, bD, glades::GMath::TANH, tc.outAct, tc.outType, tc.packing, tc.readout);
						bD[t][m].setComponent(lane, baseD);

						const double fdD = (lPD - lMD) / (2.0 * epsD);
						const double ag = static_cast<double>(analyticGB[t][m].component(lane));
						const double relErrD = fabs(fdD - ag) / std::max(1e-5, std::max(fabs(fdD), fabs(ag)));
						if (relErrD > maxRelErrD)
							maxRelErrD = relErrD;

						const float baseF = net.tensorQnn.T[t].bias[m].component(lane);
						const float pVal = baseF + epsF;
						const float mVal = baseF - epsF;
						net.tensorQnn.T[t].bias[m].setComponent(lane, pVal);
						CaptureLastEpochCb cbP;
						ASSERT("FD b+ test succeeds", net.test(&di, &cbP).ok());
						const double lPF = static_cast<double>(cbP.last.totalError) * static_cast<double>(tc.lossScaleForFd);
						net.tensorQnn.T[t].bias[m].setComponent(lane, mVal);
						CaptureLastEpochCb cbM;
						ASSERT("FD b- test succeeds", net.test(&di, &cbM).ok());
						const double lMF = static_cast<double>(cbM.last.totalError) * static_cast<double>(tc.lossScaleForFd);
						net.tensorQnn.T[t].bias[m].setComponent(lane, baseF);
						const double fdF = (lPF - lMF) / (static_cast<double>(pVal) - static_cast<double>(mVal));
						const double absErrF = fabs(fdF - ag);
						if (absErrF > maxAbsErrF)
							maxAbsErrF = absErrF;
					}
				}
			}

			// Check input activation gradients dL/dx_f against analytic delta[0].
			for (unsigned int f = 0u; f < tc.realIn; ++f)
			{
				const double baseD = xD[f];
				xD[f] = baseD + epsD;
				const double lPD = eval_qnn_loss_double(xD, yD, net.tensorQnn.qSizes, wD, bD, glades::GMath::TANH, tc.outAct, tc.outType, tc.packing, tc.readout);
				xD[f] = baseD - epsD;
				const double lMD = eval_qnn_loss_double(xD, yD, net.tensorQnn.qSizes, wD, bD, glades::GMath::TANH, tc.outAct, tc.outType, tc.packing, tc.readout);
				xD[f] = baseD;

				const double fdD = (lPD - lMD) / (2.0 * epsD);
				const double ag = (tc.packing == glades::QuaternionConfig::PACK_GROUPS_OF_4)
				    ? static_cast<double>(analyticGIn[f / 4u].component(f % 4u))
				    : static_cast<double>(analyticGIn[f].r);
				const double relErrD = fabs(fdD - ag) / std::max(1e-5, std::max(fabs(fdD), fabs(ag)));
				if (relErrD > maxRelErrD)
					maxRelErrD = relErrD;

				const float baseF = di.testMatrix[0][f];
				const float pVal = baseF + epsF;
				const float mVal = baseF - epsF;
				di.testMatrix[0][f] = pVal;
				CaptureLastEpochCb cbP;
				ASSERT("FD x+ test succeeds", net.test(&di, &cbP).ok());
				const double lPF = static_cast<double>(cbP.last.totalError) * static_cast<double>(tc.lossScaleForFd);
				di.testMatrix[0][f] = mVal;
				CaptureLastEpochCb cbM;
				ASSERT("FD x- test succeeds", net.test(&di, &cbM).ok());
				const double lMF = static_cast<double>(cbM.last.totalError) * static_cast<double>(tc.lossScaleForFd);
				di.testMatrix[0][f] = baseF;
				const double fdF = (lPF - lMF) / (static_cast<double>(pVal) - static_cast<double>(mVal));
				const double absErrF = fabs(fdF - ag);
				if (absErrF > maxAbsErrF)
					maxAbsErrF = absErrF;
			}

			printf("  [%s] eps=1e-4 max_rel_grad_err = %.6e | float32 NNetwork::test() eps=1e-3 max_abs_err = %.6e\n",
			       tc.label, maxRelErrD, maxAbsErrF);
			ASSERT("Central FD (eps=1e-4) max relative gradient error < 1e-3", maxRelErrD < 1e-3);
			ASSERT("End-to-end float32 NNetwork::test() FD (eps=1e-3) max absolute error < 5e-4", maxAbsErrF < 5e-4);
		}
	}
	printf("[PASS] Finite-difference gradient check verified.\n");

	// Loader verification (QuaternionInput for XOR & Iris)
	test_quaternion_input_loaders();

	std::vector<BenchmarkSummaryRow> summary;

	// Phase 1: XOR Gate
	test_qnn_xor(summary);

	// Phase 2: Iris Classification & Persistence
	test_qnn_iris(summary);

	printf("====================================================================================================\n");
	printf("QNN Convergence Summary (XOR & Iris)\n");
	printf("%-20s | %-20s | %5s | %6s | %10s | %10s | %10s | %9s\n",
	       "Dataset / Variant", "Quat Arch", "Seeds", "Epochs", "TrainTop1", "TestTop1", "TestMacro", "TestLoss");
	printf("----------------------------------------------------------------------------------------------------\n");
	for (size_t i = 0u; i < summary.size(); ++i)
	{
		const BenchmarkSummaryRow& r = summary[i];
		printf("%-20s | %-20s | %5u | %6u | %9.2f%% | %9.2f%% | %9.2f%% | %9.6f\n",
		       r.label.c_str(), r.arch.c_str(), r.seeds, r.epochs,
		       r.meanTrainTop1Pct, r.meanTestTop1Pct, r.meanTestMacroAccPct, r.meanTestLoss);
	}
	printf("====================================================================================================\n");
	printf("ALL QNN UNIT TESTS PASSED\n");
	printf("====================================================================================================\n");
}
