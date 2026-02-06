#ifndef GLADES_MAPPED_NUMBER_INPUT_H
#define GLADES_MAPPED_NUMBER_INPUT_H
// DataInput implementation backed by memory-mapped matrices.
//
// This supports production-scale datasets where loading all rows into RAM as a shmea::GMatrix
// is not acceptable. The training loop can consume this efficiently via get*RowView().
//
// Expected on-disk layout (directory import):
//   <dir>/
//     train.x.gcol   (features; float32 row-major)
//     train.y.gcol   (expected outputs; float32 row-major)
//     test.x.gcol    (optional)
//     test.y.gcol    (optional)
//
// Matrices use the format defined in MappedMatrix.h (magic "GLADES_GCOL_V1").
//
// NOTE:
// - This class does not attempt to fit preprocessing; it assumes preprocessing already happened.
// - OHE maps are intentionally left empty; this is a pure numeric tensor input for the model.

#include "DataInput.h"
#include "MappedMatrix.h"

#include <string>

namespace glades {

class MappedNumberInput : public DataInput
{
public:
	MappedNumberInput();
	virtual ~MappedNumberInput();

	// Import from a dataset directory. standardizeFlag is ignored (preprocessed already).
	virtual void import(shmea::GString path, int standardizeFlag = 0);
	virtual void import(const shmea::GTable& rawTable, int standardizeFlag = 0);

	virtual shmea::GVector<float> getTrainRow(unsigned int index) const;
	virtual shmea::GVector<float> getTrainExpectedRow(unsigned int index) const;
	virtual shmea::GVector<float> getTestRow(unsigned int index) const;
	virtual shmea::GVector<float> getTestExpectedRow(unsigned int index) const;

	virtual bool getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;
	virtual bool getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;
	virtual bool getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;
	virtual bool getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;

	virtual unsigned int getTrainSize() const;
	virtual unsigned int getTestSize() const;
	virtual unsigned int getFeatureCount() const;

	// Fixed-shape contract: backed by matrices with known dims.
	virtual bool hasFixedTrainRowSize() const { return featureCountCached > 0u; }
	virtual unsigned int getFixedTrainRowSize() const { return featureCountCached; }
	virtual bool hasFixedTrainExpectedRowSize() const { return expectedCountCached > 0u; }
	virtual unsigned int getFixedTrainExpectedRowSize() const { return expectedCountCached; }

	virtual int getType() const;

	// Helpers
	bool loadedOk() const { return loaded; }
	const std::string& lastError() const { return lastErr; }
	virtual NNetworkStatus getLastStatus() const { return lastImportStatus; }

private:
	void clear();
	bool openDir(const std::string& dirPath, std::string* errMsg);

	MappedFloatMatrix trainX;
	MappedFloatMatrix trainY;
	MappedFloatMatrix testX;
	MappedFloatMatrix testY;

	bool loaded;
	unsigned int featureCountCached;
	unsigned int expectedCountCached;

	mutable shmea::GVector<float> scratchRow;
	mutable shmea::GVector<float> scratchY;
	shmea::GVector<float> emptyRow;

	std::string lastErr;
	NNetworkStatus lastImportStatus;
};

} // namespace glades

#endif

