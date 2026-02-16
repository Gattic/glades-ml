// DDPDataInputWrapper: partitions sequences across DDP workers.
//
// Wraps an existing DataInput and filters sequences so that each rank
// only sees sequences assigned to it (round-robin by index).
// All data access delegates to the inner DataInput.
#ifndef _GLADES_DDP_DATA_INPUT_WRAPPER_H
#define _GLADES_DDP_DATA_INPUT_WRAPPER_H

#include "DataInput.h"
#include <vector>

namespace glades {

class DDPDataInputWrapper : public DataInput
{
	const DataInput* inner;
	int ddpRank;
	int ddpWorldSize;

	// Maps from local sequence index to global sequence index in inner.
	std::vector<unsigned int> trainSeqMap;
	std::vector<unsigned int> testSeqMap;

	void buildMaps();

	// Non-copyable.
	DDPDataInputWrapper(const DDPDataInputWrapper&);
	DDPDataInputWrapper& operator=(const DDPDataInputWrapper&);

public:
	DDPDataInputWrapper(const DataInput* innerInput, int rank, int worldSize);
	virtual ~DDPDataInputWrapper();

	virtual void import(shmea::GString, int) {}
	virtual void import(const shmea::GTable&, int) {}

	virtual shmea::GVector<float> getTrainRow(unsigned int idx) const;
	virtual shmea::GVector<float> getTrainExpectedRow(unsigned int idx) const;
	virtual shmea::GVector<float> getTestRow(unsigned int idx) const;
	virtual shmea::GVector<float> getTestExpectedRow(unsigned int idx) const;

	virtual bool getTrainRowView(unsigned int idx, const float*& outData, unsigned int& outSize) const;
	virtual bool getTrainExpectedRowView(unsigned int idx, const float*& outData, unsigned int& outSize) const;
	virtual bool getTestRowView(unsigned int idx, const float*& outData, unsigned int& outSize) const;
	virtual bool getTestExpectedRowView(unsigned int idx, const float*& outData, unsigned int& outSize) const;

	virtual bool getTrainTokenId(unsigned int idx, int& outTokenId) const;
	virtual bool getTrainExpectedTokenId(unsigned int idx, int& outTokenId) const;
	virtual bool getTestTokenId(unsigned int idx, int& outTokenId) const;
	virtual bool getTestExpectedTokenId(unsigned int idx, int& outTokenId) const;

	virtual unsigned int getTrainSize() const;
	virtual unsigned int getTestSize() const;
	virtual unsigned int getFeatureCount() const;

	virtual unsigned int getTrainSequenceCount() const;
	virtual unsigned int getTrainSequenceLength(unsigned int seqIdx) const;
	virtual shmea::GVector<float> getTrainSequenceRow(unsigned int seqIdx, unsigned int t) const;
	virtual shmea::GVector<float> getTrainSequenceExpectedRow(unsigned int seqIdx, unsigned int t) const;

	virtual unsigned int getTestSequenceCount() const;
	virtual unsigned int getTestSequenceLength(unsigned int seqIdx) const;
	virtual shmea::GVector<float> getTestSequenceRow(unsigned int seqIdx, unsigned int t) const;
	virtual shmea::GVector<float> getTestSequenceExpectedRow(unsigned int seqIdx, unsigned int t) const;

	virtual bool hasFixedTrainRowSize() const;
	virtual unsigned int getFixedTrainRowSize() const;
	virtual bool hasFixedTrainExpectedRowSize() const;
	virtual unsigned int getFixedTrainExpectedRowSize() const;

	virtual NNetworkStatus getLastStatus() const;

	virtual int getType() const;
};

} // namespace glades

#endif
