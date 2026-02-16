// DDPDataInputWrapper implementation.
//
// Partitions sequences round-robin across DDP ranks.
// Worker r gets sequences {s : s % worldSize == rank}.
#include "DDPDataInputWrapper.h"

using namespace glades;

DDPDataInputWrapper::DDPDataInputWrapper(const DataInput* innerInput, int rank, int worldSize)
    : inner(innerInput), ddpRank(rank), ddpWorldSize(worldSize)
{
	buildMaps();
}

DDPDataInputWrapper::~DDPDataInputWrapper()
{
	// inner is not owned by this wrapper.
}

void DDPDataInputWrapper::buildMaps()
{
	trainSeqMap.clear();
	testSeqMap.clear();

	if (!inner || ddpWorldSize <= 1)
		return;

	const unsigned int trainSeqCount = inner->getTrainSequenceCount();
	for (unsigned int s = 0; s < trainSeqCount; ++s)
	{
		if (static_cast<int>(s % static_cast<unsigned int>(ddpWorldSize)) == ddpRank)
			trainSeqMap.push_back(s);
	}

	const unsigned int testSeqCount = inner->getTestSequenceCount();
	for (unsigned int s = 0; s < testSeqCount; ++s)
	{
		if (static_cast<int>(s % static_cast<unsigned int>(ddpWorldSize)) == ddpRank)
			testSeqMap.push_back(s);
	}
}

// --- Train data access (row-based, delegates to inner) ---

shmea::GVector<float> DDPDataInputWrapper::getTrainRow(unsigned int idx) const
{
	if (inner) return inner->getTrainRow(idx);
	return shmea::GVector<float>();
}

shmea::GVector<float> DDPDataInputWrapper::getTrainExpectedRow(unsigned int idx) const
{
	if (inner) return inner->getTrainExpectedRow(idx);
	return shmea::GVector<float>();
}

shmea::GVector<float> DDPDataInputWrapper::getTestRow(unsigned int idx) const
{
	if (inner) return inner->getTestRow(idx);
	return shmea::GVector<float>();
}

shmea::GVector<float> DDPDataInputWrapper::getTestExpectedRow(unsigned int idx) const
{
	if (inner) return inner->getTestExpectedRow(idx);
	return shmea::GVector<float>();
}

bool DDPDataInputWrapper::getTrainRowView(unsigned int idx, const float*& outData, unsigned int& outSize) const
{
	if (inner) return inner->getTrainRowView(idx, outData, outSize);
	outData = NULL; outSize = 0; return false;
}

bool DDPDataInputWrapper::getTrainExpectedRowView(unsigned int idx, const float*& outData, unsigned int& outSize) const
{
	if (inner) return inner->getTrainExpectedRowView(idx, outData, outSize);
	outData = NULL; outSize = 0; return false;
}

bool DDPDataInputWrapper::getTestRowView(unsigned int idx, const float*& outData, unsigned int& outSize) const
{
	if (inner) return inner->getTestRowView(idx, outData, outSize);
	outData = NULL; outSize = 0; return false;
}

bool DDPDataInputWrapper::getTestExpectedRowView(unsigned int idx, const float*& outData, unsigned int& outSize) const
{
	if (inner) return inner->getTestExpectedRowView(idx, outData, outSize);
	outData = NULL; outSize = 0; return false;
}

bool DDPDataInputWrapper::getTrainTokenId(unsigned int idx, int& outTokenId) const
{
	if (inner) return inner->getTrainTokenId(idx, outTokenId);
	return false;
}

bool DDPDataInputWrapper::getTrainExpectedTokenId(unsigned int idx, int& outTokenId) const
{
	if (inner) return inner->getTrainExpectedTokenId(idx, outTokenId);
	return false;
}

bool DDPDataInputWrapper::getTestTokenId(unsigned int idx, int& outTokenId) const
{
	if (inner) return inner->getTestTokenId(idx, outTokenId);
	return false;
}

bool DDPDataInputWrapper::getTestExpectedTokenId(unsigned int idx, int& outTokenId) const
{
	if (inner) return inner->getTestExpectedTokenId(idx, outTokenId);
	return false;
}

// --- Sizes ---

unsigned int DDPDataInputWrapper::getTrainSize() const
{
	if (inner) return inner->getTrainSize();
	return 0;
}

unsigned int DDPDataInputWrapper::getTestSize() const
{
	if (inner) return inner->getTestSize();
	return 0;
}

unsigned int DDPDataInputWrapper::getFeatureCount() const
{
	if (inner) return inner->getFeatureCount();
	return 0;
}

// --- Sequence interface (partitioned) ---

unsigned int DDPDataInputWrapper::getTrainSequenceCount() const
{
	if (!inner || ddpWorldSize <= 1)
		return inner ? inner->getTrainSequenceCount() : 0u;
	return static_cast<unsigned int>(trainSeqMap.size());
}

unsigned int DDPDataInputWrapper::getTrainSequenceLength(unsigned int seqIdx) const
{
	if (!inner)
		return 0u;
	if (ddpWorldSize <= 1)
		return inner->getTrainSequenceLength(seqIdx);
	if (seqIdx >= static_cast<unsigned int>(trainSeqMap.size()))
		return 0u;
	return inner->getTrainSequenceLength(trainSeqMap[seqIdx]);
}

shmea::GVector<float> DDPDataInputWrapper::getTrainSequenceRow(unsigned int seqIdx, unsigned int t) const
{
	if (!inner)
		return shmea::GVector<float>();
	if (ddpWorldSize <= 1)
		return inner->getTrainSequenceRow(seqIdx, t);
	if (seqIdx >= static_cast<unsigned int>(trainSeqMap.size()))
		return shmea::GVector<float>();
	return inner->getTrainSequenceRow(trainSeqMap[seqIdx], t);
}

shmea::GVector<float> DDPDataInputWrapper::getTrainSequenceExpectedRow(unsigned int seqIdx, unsigned int t) const
{
	if (!inner)
		return shmea::GVector<float>();
	if (ddpWorldSize <= 1)
		return inner->getTrainSequenceExpectedRow(seqIdx, t);
	if (seqIdx >= static_cast<unsigned int>(trainSeqMap.size()))
		return shmea::GVector<float>();
	return inner->getTrainSequenceExpectedRow(trainSeqMap[seqIdx], t);
}

unsigned int DDPDataInputWrapper::getTestSequenceCount() const
{
	if (!inner || ddpWorldSize <= 1)
		return inner ? inner->getTestSequenceCount() : 0u;
	return static_cast<unsigned int>(testSeqMap.size());
}

unsigned int DDPDataInputWrapper::getTestSequenceLength(unsigned int seqIdx) const
{
	if (!inner)
		return 0u;
	if (ddpWorldSize <= 1)
		return inner->getTestSequenceLength(seqIdx);
	if (seqIdx >= static_cast<unsigned int>(testSeqMap.size()))
		return 0u;
	return inner->getTestSequenceLength(testSeqMap[seqIdx]);
}

shmea::GVector<float> DDPDataInputWrapper::getTestSequenceRow(unsigned int seqIdx, unsigned int t) const
{
	if (!inner)
		return shmea::GVector<float>();
	if (ddpWorldSize <= 1)
		return inner->getTestSequenceRow(seqIdx, t);
	if (seqIdx >= static_cast<unsigned int>(testSeqMap.size()))
		return shmea::GVector<float>();
	return inner->getTestSequenceRow(testSeqMap[seqIdx], t);
}

shmea::GVector<float> DDPDataInputWrapper::getTestSequenceExpectedRow(unsigned int seqIdx, unsigned int t) const
{
	if (!inner)
		return shmea::GVector<float>();
	if (ddpWorldSize <= 1)
		return inner->getTestSequenceExpectedRow(seqIdx, t);
	if (seqIdx >= static_cast<unsigned int>(testSeqMap.size()))
		return shmea::GVector<float>();
	return inner->getTestSequenceExpectedRow(testSeqMap[seqIdx], t);
}

// --- Fixed-size contracts (delegate to inner) ---

bool DDPDataInputWrapper::hasFixedTrainRowSize() const
{
	return inner ? inner->hasFixedTrainRowSize() : false;
}

unsigned int DDPDataInputWrapper::getFixedTrainRowSize() const
{
	return inner ? inner->getFixedTrainRowSize() : 0u;
}

bool DDPDataInputWrapper::hasFixedTrainExpectedRowSize() const
{
	return inner ? inner->hasFixedTrainExpectedRowSize() : false;
}

unsigned int DDPDataInputWrapper::getFixedTrainExpectedRowSize() const
{
	return inner ? inner->getFixedTrainExpectedRowSize() : 0u;
}

NNetworkStatus DDPDataInputWrapper::getLastStatus() const
{
	return inner ? inner->getLastStatus() : NNetworkStatus();
}

int DDPDataInputWrapper::getType() const
{
	return inner ? inner->getType() : DataInput::CSV;
}
