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
#include "QuaternionInput.h"
#include "../rng.h"

#include <cmath>

namespace {

static glades::Quaternion sample_unit_quaternion(glades::rng::Engine& eng)
{
	for (unsigned int tries = 0u; tries < 64u; ++tries)
	{
		glades::Quaternion q(
		    glades::rng::standard_normal(eng),
		    glades::rng::standard_normal(eng),
		    glades::rng::standard_normal(eng),
		    glades::rng::standard_normal(eng));
		const float n = q.norm();
		if (n > 1e-6f)
		{
			q /= n;
			if (q.r < 0.0f)
				q = -q;
			return q;
		}
	}
	return glades::Quaternion::one();
}

static glades::Quaternion sample_unit_vec3_pure(glades::rng::Engine& eng)
{
	for (unsigned int tries = 0u; tries < 64u; ++tries)
	{
		const float x = glades::rng::standard_normal(eng);
		const float y = glades::rng::standard_normal(eng);
		const float z = glades::rng::standard_normal(eng);
		const float n = sqrtf((x * x) + (y * y) + (z * z));
		if (n > 1e-6f)
			return glades::Quaternion(0.0f, x / n, y / n, z / n);
	}
	return glades::Quaternion(0.0f, 1.0f, 0.0f, 0.0f);
}

static glades::Quaternion sample_axis_angle_quaternion(glades::rng::Engine& eng,
                                                       float minAngleRad,
                                                       float maxAngleRad)
{
	const glades::Quaternion axis = sample_unit_vec3_pure(eng);
	const float theta = glades::rng::uniform_float(eng, minAngleRad, maxAngleRad);
	const float half = 0.5f * theta;
	const float c = cosf(half);
	const float s = sinf(half);
	glades::Quaternion q(c, axis.i * s, axis.j * s, axis.k * s);
	if (q.r < 0.0f)
		q = -q;
	return q;
}

static glades::Quaternion sample_hemisphere_vec3(glades::rng::Engine& eng,
                                                 bool upperHemisphere,
                                                 float minAbsZ)
{
	for (unsigned int tries = 0u; tries < 256u; ++tries)
	{
		glades::Quaternion dir = sample_unit_vec3_pure(eng);
		if (upperHemisphere && dir.k > minAbsZ)
			return dir;
		if (!upperHemisphere && dir.k < -minAbsZ)
			return dir;
	}
	return upperHemisphere
	    ? glades::Quaternion(0.0f, 0.0f, 0.0f, 1.0f)
	    : glades::Quaternion(0.0f, 0.0f, 0.0f, -1.0f);
}

static glades::Quaternion sample_signed_k_unit_quaternion(glades::rng::Engine& eng,
                                                          bool positiveK,
                                                          float minAbsK)
{
	for (unsigned int tries = 0u; tries < 256u; ++tries)
	{
		glades::Quaternion q = sample_unit_quaternion(eng);
		if (positiveK && q.k > minAbsK)
			return q;
		if (!positiveK && q.k < -minAbsK)
			return q;
	}
	return positiveK
	    ? glades::Quaternion(0.70710678f, 0.0f, 0.0f, 0.70710678f)
	    : glades::Quaternion(0.70710678f, 0.0f, 0.0f, -0.70710678f);
}

} // namespace

namespace glades {

QuaternionInput::QuaternionInput()
    : NumberInput(),
      inputPackingMode(QUAT_PACK_GROUPS_OF_4),
      outputReadoutMode(QUAT_READOUT_REAL_PART),
      quatInputCount(0u),
      quatOutputCount(0u),
      realOutputCount(0u),
      lastStatus()
{
}

QuaternionInput::~QuaternionInput()
{
	clearQuaternionBuffers();
}

void QuaternionInput::clearQuaternionBuffers()
{
	quatInputCount = 0u;
	quatOutputCount = 0u;
	realOutputCount = 0u;
	trainQuatInputs.clear();
	trainQuatTargets.clear();
	testQuatInputs.clear();
	testQuatTargets.clear();
	classLabels.clear();
}

bool QuaternionInput::rebuildQuaternionBuffers(QuaternionInputPackingMode packing,
                                               QuaternionOutputReadoutMode readout)
{
	inputPackingMode = packing;
	outputReadoutMode = readout;
	trainQuatInputs.clear();
	trainQuatTargets.clear();
	testQuatInputs.clear();
	testQuatTargets.clear();
	quatInputCount = 0u;
	quatOutputCount = 0u;
	realOutputCount = 0u;

	const unsigned int nTrain = getTrainSize();
	const unsigned int nTest = getTestSize();
	const unsigned int realIn = getFeatureCount();
	unsigned int realOut = getFixedTrainExpectedRowSize();
	if (realOut == 0u && nTest > 0u)
	{
		const float* yPtr = NULL;
		unsigned int ySz = 0u;
		if (getTestExpectedRowView(0u, yPtr, ySz))
			realOut = ySz;
	}

	if (realIn == 0u || realOut == 0u || (nTrain == 0u && nTest == 0u))
	{
		lastStatus = NNetworkStatus(NNetworkStatus::EMPTY_DATA,
		                            "QuaternionInput::rebuildQuaternionBuffers: empty dataset or zero feature/output width");
		return false;
	}

	const unsigned int qIn = quaternionInputCount(realIn, packing);
	const unsigned int qOut = quaternionOutputCount(realOut, readout);
	if (qIn == 0u || qOut == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::rebuildQuaternionBuffers: invalid quaternion dimensions");
		return false;
	}

	quatInputCount = qIn;
	quatOutputCount = qOut;
	realOutputCount = realOut;

	trainQuatInputs.assign(static_cast<size_t>(nTrain) * static_cast<size_t>(qIn), Quaternion::zero());
	trainQuatTargets.assign(static_cast<size_t>(nTrain) * static_cast<size_t>(qOut), Quaternion::zero());
	for (unsigned int r = 0u; r < nTrain; ++r)
	{
		const float* xPtr = NULL;
		unsigned int xSz = 0u;
		if (!getTrainRowView(r, xPtr, xSz))
		{
			clearQuaternionBuffers();
			lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
			                            "QuaternionInput::rebuildQuaternionBuffers: failed to read train input row");
			return false;
		}
		Quaternion* dstIn = &trainQuatInputs[static_cast<size_t>(r) * static_cast<size_t>(qIn)];
		encodeRealToQuaternions(xPtr, xSz, realIn, packing, dstIn, qIn);
		for (unsigned int q = 0u; q < qIn; ++q)
		{
			if (!dstIn[q].isFinite())
			{
				clearQuaternionBuffers();
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
				                            "QuaternionInput::rebuildQuaternionBuffers: non-finite train feature value");
				return false;
			}
		}

		const float* yPtr = NULL;
		unsigned int ySz = 0u;
		if (!getTrainExpectedRowView(r, yPtr, ySz))
		{
			clearQuaternionBuffers();
			lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
			                            "QuaternionInput::rebuildQuaternionBuffers: failed to read train expected row");
			return false;
		}
		Quaternion* dstOut = &trainQuatTargets[static_cast<size_t>(r) * static_cast<size_t>(qOut)];
		encodeTargetsToQuaternions(yPtr, ySz, realOut, readout, dstOut, qOut);
		for (unsigned int q = 0u; q < qOut; ++q)
		{
			if (!dstOut[q].isFinite())
			{
				clearQuaternionBuffers();
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
				                            "QuaternionInput::rebuildQuaternionBuffers: non-finite train target value");
				return false;
			}
		}
	}

	testQuatInputs.assign(static_cast<size_t>(nTest) * static_cast<size_t>(qIn), Quaternion::zero());
	testQuatTargets.assign(static_cast<size_t>(nTest) * static_cast<size_t>(qOut), Quaternion::zero());
	for (unsigned int r = 0u; r < nTest; ++r)
	{
		const float* xPtr = NULL;
		unsigned int xSz = 0u;
		if (!getTestRowView(r, xPtr, xSz))
		{
			clearQuaternionBuffers();
			lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
			                            "QuaternionInput::rebuildQuaternionBuffers: failed to read test input row");
			return false;
		}
		Quaternion* dstIn = &testQuatInputs[static_cast<size_t>(r) * static_cast<size_t>(qIn)];
		encodeRealToQuaternions(xPtr, xSz, realIn, packing, dstIn, qIn);
		for (unsigned int q = 0u; q < qIn; ++q)
		{
			if (!dstIn[q].isFinite())
			{
				clearQuaternionBuffers();
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
				                            "QuaternionInput::rebuildQuaternionBuffers: non-finite test feature value");
				return false;
			}
		}

		const float* yPtr = NULL;
		unsigned int ySz = 0u;
		if (!getTestExpectedRowView(r, yPtr, ySz))
		{
			clearQuaternionBuffers();
			lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
			                            "QuaternionInput::rebuildQuaternionBuffers: failed to read test expected row");
			return false;
		}
		Quaternion* dstOut = &testQuatTargets[static_cast<size_t>(r) * static_cast<size_t>(qOut)];
		encodeTargetsToQuaternions(yPtr, ySz, realOut, readout, dstOut, qOut);
		for (unsigned int q = 0u; q < qOut; ++q)
		{
			if (!dstOut[q].isFinite())
			{
				clearQuaternionBuffers();
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
				                            "QuaternionInput::rebuildQuaternionBuffers: non-finite test target value");
				return false;
			}
		}
	}

	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadTabularWithSplit(const shmea::GTable& rawTable,
                                           const TrainTestSplitConfig& splitCfg,
                                           int standardizeFlag,
                                           QuaternionInputPackingMode packing,
                                           QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (!importWithSplit(rawTable, splitCfg, standardizeFlag))
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadTabularWithSplit: NumberInput::importWithSplit failed");
		return false;
	}
	return rebuildQuaternionBuffers(packing, readout);
}

bool QuaternionInput::loadIris(const std::string& filePath,
                               const TrainTestSplitConfig& splitCfg,
                               int standardizeFlag,
                               QuaternionInputPackingMode packing,
                               QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (filePath.empty())
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadIris: empty file path");
		return false;
	}

	shmea::GTable tbl(shmea::GString(filePath.c_str()), ',', shmea::GTable::TYPE_FILE);
	if (tbl.numberOfRows() == 0u || tbl.numberOfCols() != 5u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadIris: expected 5-column Iris CSV (4 features + 1 class)");
		return false;
	}
	tbl.toggleOutput(4u);

	if (!loadTabularWithSplit(tbl, splitCfg, standardizeFlag, packing, readout))
		return false;

	if (getFeatureCount() != 4u || getRealOutputCount() != 3u)
	{
		clearQuaternionBuffers();
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadIris: expected 4 numeric features and 3 encoded classes");
		return false;
	}

	classLabels.clear();
	if (trainingOHEMaps.size() > 4u && trainingOHEMaps[4u])
	{
		const unsigned int nClasses = trainingOHEMaps[4u]->size();
		classLabels.reserve(nClasses);
		for (unsigned int c = 0u; c < nClasses; ++c)
			classLabels.push_back(std::string(trainingOHEMaps[4u]->classAt(c).c_str()));
	}

	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadXOR(const std::string& filePath,
                              QuaternionInputPackingMode packing,
                              QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (filePath.empty())
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadXOR: empty file path");
		return false;
	}

	shmea::GTable tbl(shmea::GString(filePath.c_str()), ',', shmea::GTable::TYPE_FILE);
	if (tbl.numberOfRows() != 4u || tbl.numberOfCols() != 3u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadXOR: expected 4x3 XOR truth table CSV");
		return false;
	}

	trainMatrix = shmea::GMatrix(4, shmea::GVector<float>(2, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(4, shmea::GVector<float>(1, 0.0f));
	for (unsigned int r = 0u; r < 4u; ++r)
	{
		trainMatrix[r][0] = tbl.getCell(r, 0u).getFloat();
		trainMatrix[r][1] = tbl.getCell(r, 1u).getFloat();
		trainExpectedMatrix[r][0] = tbl.getCell(r, 2u).getFloat();
	}
	testMatrix = trainMatrix;
	testExpectedMatrix = trainExpectedMatrix;
	featureCountCached = 2u;
	expectedCountCached = 1u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	classLabels.push_back("0");
	classLabels.push_back("1");
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadParityFile(const std::string& filePath,
                                     QuaternionInputPackingMode packing,
                                     QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (filePath.empty())
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadParityFile: empty file path");
		return false;
	}

	shmea::GTable tbl(shmea::GString(filePath.c_str()), ',', shmea::GTable::TYPE_FILE);
	const unsigned int nRows = tbl.numberOfRows();
	const unsigned int nCols = tbl.numberOfCols();
	if (nRows == 0u || nCols < 2u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadParityFile: expected >= 1 row and >= 2 columns");
		return false;
	}

	const unsigned int nIn = nCols - 1u;
	trainMatrix = shmea::GMatrix(nRows, shmea::GVector<float>(nIn, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(nRows, shmea::GVector<float>(1u, 0.0f));
	for (unsigned int r = 0u; r < nRows; ++r)
	{
		for (unsigned int c = 0u; c < nIn; ++c)
			trainMatrix[r][c] = tbl.getCell(r, c).getFloat();
		trainExpectedMatrix[r][0] = tbl.getCell(r, nIn).getFloat();
	}
	testMatrix = trainMatrix;
	testExpectedMatrix = trainExpectedMatrix;
	featureCountCached = nIn;
	expectedCountCached = 1u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	classLabels.push_back("0");
	classLabels.push_back("1");
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadBinaryParity(unsigned int nBits,
                                       QuaternionInputPackingMode packing,
                                       QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (nBits == 0u || nBits > 8u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadBinaryParity: nBits must be in [1, 8]");
		return false;
	}

	const unsigned int nRows = (1u << nBits);
	trainMatrix = shmea::GMatrix(nRows, shmea::GVector<float>(nBits, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(nRows, shmea::GVector<float>(1u, 0.0f));
	for (unsigned int mask = 0u; mask < nRows; ++mask)
	{
		unsigned int parity = 0u;
		for (unsigned int b = 0u; b < nBits; ++b)
		{
			const unsigned int bit = (mask >> (nBits - 1u - b)) & 1u;
			trainMatrix[mask][b] = static_cast<float>(bit);
			parity ^= bit;
		}
		trainExpectedMatrix[mask][0] = static_cast<float>(parity);
	}
	testMatrix = trainMatrix;
	testExpectedMatrix = trainExpectedMatrix;
	featureCountCached = nBits;
	expectedCountCached = 1u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	classLabels.push_back("0");
	classLabels.push_back("1");
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadWine(const std::string& filePath,
                               const TrainTestSplitConfig& splitCfg,
                               int standardizeFlag,
                               QuaternionInputPackingMode packing,
                               QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (filePath.empty())
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadWine: empty file path");
		return false;
	}

	shmea::GTable tbl(shmea::GString(filePath.c_str()), ',', shmea::GTable::TYPE_FILE);
	if (tbl.numberOfRows() == 0u || tbl.numberOfCols() != 14u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadWine: expected 14-column Wine CSV (13 features + 1 class)");
		return false;
	}
	tbl.toggleOutput(13u);

	if (!loadTabularWithSplit(tbl, splitCfg, standardizeFlag, packing, readout))
		return false;

	if (getFeatureCount() != 13u || getRealOutputCount() != 3u)
	{
		clearQuaternionBuffers();
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadWine: expected 13 numeric features and 3 encoded classes");
		return false;
	}

	classLabels.clear();
	if (trainingOHEMaps.size() > 13u && trainingOHEMaps[13u])
	{
		const unsigned int nClasses = trainingOHEMaps[13u]->size();
		classLabels.reserve(nClasses);
		for (unsigned int c = 0u; c < nClasses; ++c)
			classLabels.push_back(std::string(trainingOHEMaps[13u]->classAt(c).c_str()));
	}

	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadDowJonesOHLC(const std::string& trainPath,
                                       const std::string& testPath,
                                       float* outTargetMean,
                                       float* outTargetStd,
                                       QuaternionInputPackingMode packing,
                                       QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (trainPath.empty() || testPath.empty())
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadDowJonesOHLC: empty train/test path");
		return false;
	}

	shmea::GTable trainTbl(shmea::GString(trainPath.c_str()), ',', shmea::GTable::TYPE_FILE);
	shmea::GTable testTbl(shmea::GString(testPath.c_str()), ',', shmea::GTable::TYPE_FILE);
	if (trainTbl.numberOfRows() == 0u || testTbl.numberOfRows() == 0u ||
	    trainTbl.numberOfCols() < 13u || testTbl.numberOfCols() < 13u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadDowJonesOHLC: invalid Dow Jones CSV dimensions");
		return false;
	}

	// Columns in dow_jones_index_{train,test}.csv:
	//   3=open, 4=high, 5=low, 6=close, 12=next_weeks_close.
	const unsigned int ohlcCols[4] = { 3u, 4u, 5u, 6u };
	const unsigned int targetCol = 12u;
	const unsigned int nTrain = trainTbl.numberOfRows();
	const unsigned int nTest = testTbl.numberOfRows();

	double mean[4] = { 0.0, 0.0, 0.0, 0.0 };
	double var[4] = { 0.0, 0.0, 0.0, 0.0 };
	for (unsigned int r = 0u; r < nTrain; ++r)
	{
		for (unsigned int c = 0u; c < 4u; ++c)
			mean[c] += static_cast<double>(trainTbl.getCell(r, ohlcCols[c]).getFloat());
	}
	for (unsigned int c = 0u; c < 4u; ++c)
		mean[c] /= static_cast<double>(nTrain);

	for (unsigned int r = 0u; r < nTrain; ++r)
	{
		for (unsigned int c = 0u; c < 4u; ++c)
		{
			const double d = static_cast<double>(trainTbl.getCell(r, ohlcCols[c]).getFloat()) - mean[c];
			var[c] += d * d;
		}
	}

	double stdv[4] = { 1.0, 1.0, 1.0, 1.0 };
	for (unsigned int c = 0u; c < 4u; ++c)
	{
		const double s = sqrt(var[c] / static_cast<double>(nTrain));
		stdv[c] = (s > 1e-8) ? s : 1.0;
	}

	const double tMean = mean[3];
	const double tStd = stdv[3];
	if (outTargetMean)
		*outTargetMean = static_cast<float>(tMean);
	if (outTargetStd)
		*outTargetStd = static_cast<float>(tStd);

	trainMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(4u, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(1u, 0.0f));
	for (unsigned int r = 0u; r < nTrain; ++r)
	{
		for (unsigned int c = 0u; c < 4u; ++c)
		{
			const double v = static_cast<double>(trainTbl.getCell(r, ohlcCols[c]).getFloat());
			trainMatrix[r][c] = static_cast<float>((v - mean[c]) / stdv[c]);
		}
		const double y = static_cast<double>(trainTbl.getCell(r, targetCol).getFloat());
		trainExpectedMatrix[r][0] = static_cast<float>((y - tMean) / tStd);
	}

	testMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(4u, 0.0f));
	testExpectedMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(1u, 0.0f));
	for (unsigned int r = 0u; r < nTest; ++r)
	{
		for (unsigned int c = 0u; c < 4u; ++c)
		{
			const double v = static_cast<double>(testTbl.getCell(r, ohlcCols[c]).getFloat());
			testMatrix[r][c] = static_cast<float>((v - mean[c]) / stdv[c]);
		}
		const double y = static_cast<double>(testTbl.getCell(r, targetCol).getFloat());
		testExpectedMatrix[r][0] = static_cast<float>((y - tMean) / tStd);
	}

	featureCountCached = 4u;
	expectedCountCached = 1u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadSO3Rotation(unsigned int nTrain,
                                      unsigned int nTest,
                                      unsigned int seed,
                                      bool oodHemisphereAndAngle,
                                      QuaternionInputPackingMode packing,
                                      QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (nTrain == 0u || nTest == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadSO3Rotation: nTrain and nTest must be > 0");
		return false;
	}

	static const float kPi = 3.14159265358979323846f;
	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, static_cast<uint64_t>(seed));

	trainMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(8u, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(4u, 0.0f));
	for (unsigned int r = 0u; r < nTrain; ++r)
	{
		const float radius = glades::rng::uniform_float(eng, 0.5f, 1.5f);
		const Quaternion vDir = oodHemisphereAndAngle
		    ? sample_hemisphere_vec3(eng, true, 0.25f)
		    : sample_unit_vec3_pure(eng);
		const Quaternion v(0.0f, vDir.i * radius, vDir.j * radius, vDir.k * radius);
		const Quaternion qRot = oodHemisphereAndAngle
		    ? sample_axis_angle_quaternion(eng, 0.0f, 0.5f * kPi)
		    : sample_unit_quaternion(eng);
		const Quaternion vRot = Quaternion::rotateVec(qRot, v);

		trainMatrix[r][0] = 0.0f;
		trainMatrix[r][1] = v.i;
		trainMatrix[r][2] = v.j;
		trainMatrix[r][3] = v.k;
		trainMatrix[r][4] = qRot.r;
		trainMatrix[r][5] = qRot.i;
		trainMatrix[r][6] = qRot.j;
		trainMatrix[r][7] = qRot.k;

		trainExpectedMatrix[r][0] = 0.0f;
		trainExpectedMatrix[r][1] = vRot.i;
		trainExpectedMatrix[r][2] = vRot.j;
		trainExpectedMatrix[r][3] = vRot.k;
	}

	testMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(8u, 0.0f));
	testExpectedMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(4u, 0.0f));
	for (unsigned int r = 0u; r < nTest; ++r)
	{
		const float radius = glades::rng::uniform_float(eng, 0.5f, 1.5f);
		const Quaternion vDir = oodHemisphereAndAngle
		    ? sample_hemisphere_vec3(eng, false, 0.25f)
		    : sample_unit_vec3_pure(eng);
		const Quaternion v(0.0f, vDir.i * radius, vDir.j * radius, vDir.k * radius);
		const Quaternion qRot = oodHemisphereAndAngle
		    ? sample_axis_angle_quaternion(eng, 0.5f * kPi, kPi)
		    : sample_unit_quaternion(eng);
		const Quaternion vRot = Quaternion::rotateVec(qRot, v);

		testMatrix[r][0] = 0.0f;
		testMatrix[r][1] = v.i;
		testMatrix[r][2] = v.j;
		testMatrix[r][3] = v.k;
		testMatrix[r][4] = qRot.r;
		testMatrix[r][5] = qRot.i;
		testMatrix[r][6] = qRot.j;
		testMatrix[r][7] = qRot.k;

		testExpectedMatrix[r][0] = 0.0f;
		testExpectedMatrix[r][1] = vRot.i;
		testExpectedMatrix[r][2] = vRot.j;
		testExpectedMatrix[r][3] = vRot.k;
	}

	featureCountCached = 8u;
	expectedCountCached = 4u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::loadSO3Composition(unsigned int nTrain,
                                         unsigned int nTest,
                                         unsigned int seed,
                                         bool oodRotatedFrame,
                                         QuaternionInputPackingMode packing,
                                         QuaternionOutputReadoutMode readout)
{
	clearQuaternionBuffers();
	if (nTrain == 0u || nTest == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		                            "QuaternionInput::loadSO3Composition: nTrain and nTest must be > 0");
		return false;
	}

	glades::rng::Engine eng;
	glades::rng::seed_engine(eng, static_cast<uint64_t>(seed));

	trainMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(8u, 0.0f));
	trainExpectedMatrix = shmea::GMatrix(nTrain, shmea::GVector<float>(4u, 0.0f));
	for (unsigned int r = 0u; r < nTrain; ++r)
	{
		const Quaternion q1 = oodRotatedFrame
		    ? sample_signed_k_unit_quaternion(eng, true, 0.15f)
		    : sample_unit_quaternion(eng);
		const Quaternion q2 = oodRotatedFrame
		    ? sample_signed_k_unit_quaternion(eng, true, 0.15f)
		    : sample_unit_quaternion(eng);
		const Quaternion qOut = Quaternion::mul(q1, q2);

		trainMatrix[r][0] = q1.r;
		trainMatrix[r][1] = q1.i;
		trainMatrix[r][2] = q1.j;
		trainMatrix[r][3] = q1.k;
		trainMatrix[r][4] = q2.r;
		trainMatrix[r][5] = q2.i;
		trainMatrix[r][6] = q2.j;
		trainMatrix[r][7] = q2.k;

		trainExpectedMatrix[r][0] = qOut.r;
		trainExpectedMatrix[r][1] = qOut.i;
		trainExpectedMatrix[r][2] = qOut.j;
		trainExpectedMatrix[r][3] = qOut.k;
	}

	testMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(8u, 0.0f));
	testExpectedMatrix = shmea::GMatrix(nTest, shmea::GVector<float>(4u, 0.0f));
	for (unsigned int r = 0u; r < nTest; ++r)
	{
		const Quaternion q1 = oodRotatedFrame
		    ? sample_signed_k_unit_quaternion(eng, false, 0.15f)
		    : sample_unit_quaternion(eng);
		const Quaternion q2 = oodRotatedFrame
		    ? sample_signed_k_unit_quaternion(eng, false, 0.15f)
		    : sample_unit_quaternion(eng);
		const Quaternion qOut = Quaternion::mul(q1, q2);

		testMatrix[r][0] = q1.r;
		testMatrix[r][1] = q1.i;
		testMatrix[r][2] = q1.j;
		testMatrix[r][3] = q1.k;
		testMatrix[r][4] = q2.r;
		testMatrix[r][5] = q2.i;
		testMatrix[r][6] = q2.j;
		testMatrix[r][7] = q2.k;

		testExpectedMatrix[r][0] = qOut.r;
		testExpectedMatrix[r][1] = qOut.i;
		testExpectedMatrix[r][2] = qOut.j;
		testExpectedMatrix[r][3] = qOut.k;
	}

	featureCountCached = 8u;
	expectedCountCached = 4u;
	loaded = true;

	if (!rebuildQuaternionBuffers(packing, readout))
		return false;

	classLabels.clear();
	lastStatus = NNetworkStatus();
	return true;
}

bool QuaternionInput::getTrainQuatInputRow(unsigned int index, const Quaternion*& outData, unsigned int& outCount) const
{
	const unsigned int nTrain = getTrainSize();
	if (index >= nTrain || quatInputCount == 0u)
		return false;
	const size_t offset = static_cast<size_t>(index) * static_cast<size_t>(quatInputCount);
	if (offset + static_cast<size_t>(quatInputCount) > trainQuatInputs.size())
		return false;
	outData = &trainQuatInputs[offset];
	outCount = quatInputCount;
	return true;
}

bool QuaternionInput::getTrainQuatTargetRow(unsigned int index, const Quaternion*& outData, unsigned int& outCount) const
{
	const unsigned int nTrain = getTrainSize();
	if (index >= nTrain || quatOutputCount == 0u)
		return false;
	const size_t offset = static_cast<size_t>(index) * static_cast<size_t>(quatOutputCount);
	if (offset + static_cast<size_t>(quatOutputCount) > trainQuatTargets.size())
		return false;
	outData = &trainQuatTargets[offset];
	outCount = quatOutputCount;
	return true;
}

bool QuaternionInput::getTestQuatInputRow(unsigned int index, const Quaternion*& outData, unsigned int& outCount) const
{
	const unsigned int nTest = getTestSize();
	if (index >= nTest || quatInputCount == 0u)
		return false;
	const size_t offset = static_cast<size_t>(index) * static_cast<size_t>(quatInputCount);
	if (offset + static_cast<size_t>(quatInputCount) > testQuatInputs.size())
		return false;
	outData = &testQuatInputs[offset];
	outCount = quatInputCount;
	return true;
}

bool QuaternionInput::getTestQuatTargetRow(unsigned int index, const Quaternion*& outData, unsigned int& outCount) const
{
	const unsigned int nTest = getTestSize();
	if (index >= nTest || quatOutputCount == 0u)
		return false;
	const size_t offset = static_cast<size_t>(index) * static_cast<size_t>(quatOutputCount);
	if (offset + static_cast<size_t>(quatOutputCount) > testQuatTargets.size())
		return false;
	outData = &testQuatTargets[offset];
	outCount = quatOutputCount;
	return true;
}

} // namespace glades
