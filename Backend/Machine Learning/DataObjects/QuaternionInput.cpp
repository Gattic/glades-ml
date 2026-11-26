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
