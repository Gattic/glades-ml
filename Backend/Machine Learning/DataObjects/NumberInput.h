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
#ifndef _GNUMBERINPUT
#define _GNUMBERINPUT

#include "DataInput.h"
#include "tabular_preprocessing.h"
#include "Backend/Database/GString.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/image.h"
#include <stdio.h>
#include <vector>
#include <map>
#include <string>
#include <stdint.h>

namespace glades 
{

class NumberInput : public DataInput
{
public:
	// Explicit train/test split configuration for tabular datasets.
	// This is used by importWithSplit() to deterministically produce a test set while
	// fitting preprocessing (OHE/scalers) on TRAIN ONLY (prevents leakage).
	struct TrainTestSplitConfig
	{
		float testFraction;  // in [0,1]
		bool shuffle;        // shuffle row order before splitting
		bool stratify;       // best-effort stratify by a single output column
		uint64_t seed;       // deterministic seed for shuffle/stratify

		TrainTestSplitConfig()
		    : testFraction(0.2f),
		      shuffle(true),
		      stratify(true),
		      seed(1u)
		{
		}
	};

	// Sparse row representation for high-cardinality categorical inputs.
	// Centralized in tabular_preprocessing.h so both NumberInput and CV share the same definition.
	typedef glades::TabularSparseRow SparseRow;

	// Path, Label
	shmea::GMatrix trainMatrix;
	shmea::GMatrix trainExpectedMatrix;
	shmea::GMatrix testMatrix;
	shmea::GMatrix testExpectedMatrix;

	// === Packed contiguous dense storage (optional) ===
	//
	// Motivation:
	// - shmea::GMatrix is a "vector of rows", where each row owns a separate heap buffer.
	//   For hot training paths that repeatedly iterate over rows, this adds pointer chasing
	//   and cache misses compared to a single contiguous float array.
	//
	// Design:
	// - When enabled, NumberInput maintains an additional packed row-major buffer for each
	//   dense matrix: X/Y for train/test.
	// - get*RowView() will prefer these packed buffers when present, providing a stable
	//   pointer to contiguous floats.
	// - Callers can choose to keep or drop the dense GMatrix copies to trade memory for
	//   compatibility/debuggability.
	bool contiguousDenseEnabled;
	bool keepDenseMatrixInContiguousMode;
	unsigned int trainRowsCachedDense;
	unsigned int testRowsCachedDense;
	std::vector<float> trainXFlat;
	std::vector<float> trainYFlat;
	std::vector<float> testXFlat;
	std::vector<float> testYFlat;

	// Optional sparse storage for INPUT feature rows (enabled via enableSparseInput()).
	std::vector<SparseRow> trainSparseRows;
	std::vector<SparseRow> testSparseRows;

	shmea::GVector<float> emptyRow;
	shmea::GString name;
	bool loaded;

	// Sparse mode controls.
	// - When enabled, categorical INPUT features are encoded as true one-hot (1.0/0.0) in sparse form.
	// - Numeric features are always stored explicitly.
	// - If keepDense==true, dense matrices are also populated (debug/compatibility).
	bool sparseInputEnabled;
	bool keepDenseMatrixInSparseMode;

	// Cached shapes (so we can run with sparse-only storage).
	unsigned int featureCountCached;
	unsigned int expectedCountCached;

	// Scratch dense row for callers that still request a dense row in sparse mode.
	mutable shmea::GVector<float> denseScratchRow;

	NumberInput() :
	    contiguousDenseEnabled(false),
	    keepDenseMatrixInContiguousMode(true),
	    trainRowsCachedDense(0u),
	    testRowsCachedDense(0u),
	    name(""),
	    loaded(false),
	    sparseInputEnabled(false),
	    keepDenseMatrixInSparseMode(false),
	    featureCountCached(0u),
	    expectedCountCached(0u)
	{
	    trainingOHEMaps.clear();
	    testingOHEMaps.clear();
	    trainingFeatureIsCategorical.clear();
	    testingFeatureIsCategorical.clear();
	    trainMatrix.clear();
	    trainExpectedMatrix.clear();
	    testMatrix.clear();
	    testExpectedMatrix.clear();
	    trainXFlat.clear();
	    trainYFlat.clear();
	    testXFlat.clear();
	    testYFlat.clear();
	    trainSparseRows.clear();
	    testSparseRows.clear();
	    denseScratchRow.clear();
	}

	virtual ~NumberInput()
	{
	    name = "";
	    loaded = false;
	    trainingOHEMaps.clear();
	    testingOHEMaps.clear();
	    trainingFeatureIsCategorical.clear();
	    testingFeatureIsCategorical.clear();
	    trainMatrix.clear();
	    trainExpectedMatrix.clear();
	    testMatrix.clear();
	    testExpectedMatrix.clear();
	    trainXFlat.clear();
	    trainYFlat.clear();
	    testXFlat.clear();
	    testYFlat.clear();
	    trainRowsCachedDense = 0u;
	    testRowsCachedDense = 0u;
	    trainSparseRows.clear();
	    testSparseRows.clear();
	    denseScratchRow.clear();
	    featureCountCached = 0u;
	    expectedCountCached = 0u;
	    sparseInputEnabled = false;
	    keepDenseMatrixInSparseMode = false;
	    contiguousDenseEnabled = false;
	    keepDenseMatrixInContiguousMode = true;
	}

	// Enable sparse storage for input feature rows.
	// IMPORTANT: This changes categorical INPUT encoding semantics to strict one-hot (1/0)
	// when sparseInputEnabled==true.
	void enableSparseInput(bool enable, bool keepDense = false)
	{
		sparseInputEnabled = enable;
		keepDenseMatrixInSparseMode = keepDense;
	}

	// Enable contiguous packed dense storage for train/test matrices.
	//
	// - enable=true: build and use packed row-major buffers for dense matrices.
	// - keepDense=true: keep trainMatrix/testMatrix and expected matrices populated too.
	//   keepDense=false: free the dense GMatrix buffers after packing (saves RAM).
	//
	// NOTE:
	// - This is orthogonal to sparseInputEnabled. Sparse mode affects input encoding
	//   (categoricals become true one-hot sparse). Packed dense affects *storage layout*
	//   of dense matrices when they exist.
	void enableContiguousDense(bool enable, bool keepDense = true)
	{
		contiguousDenseEnabled = enable;
		keepDenseMatrixInContiguousMode = keepDense;
	}

	// Export the currently-loaded (already preprocessed/encoded) matrices to a memory-mapped
	// dataset directory. This enables production-scale workflows where training/inference
	// reads inputs via mmap without loading full matrices into RAM.
	//
	// Layout:
	//   <dir>/
	//     train.x.gcol
	//     train.y.gcol
	//     test.x.gcol   (only if test split exists)
	//     test.y.gcol   (only if test split exists)
	//
	// Returns false on failure; errMsg (optional) describes the reason.
	bool exportMappedDataset(const std::string& dirPath, std::string* errMsg = NULL) const;

	virtual void import(shmea::GString, int = 0);
	virtual void import(const shmea::GTable&, int = 0);
	// Import explicit train/test splits (prevents preprocessing leakage).
	// NOTE: Callers must ensure output columns are configured consistently on both tables.
	bool importTrainTest(const shmea::GTable& trainRaw, const shmea::GTable& testRaw, int standardizeFlag = 0);
	// Convenience: load `trainFile` + `testFile` from disk.
	// IMPORTANT: output columns are NOT auto-detected; callers should prefer the GTable overload,
	// or set output columns on the loaded tables before calling importTrainTest(GTable,GTable,...).
	bool importTrainTest(shmea::GString trainFile, shmea::GString testFile, int standardizeFlag = 0);
	// Import a single table and create a deterministic train/test split.
	// This fits OHE/scalers on TRAIN ONLY and applies to both splits.
	bool importWithSplit(const shmea::GTable& rawTable, const TrainTestSplitConfig& cfg, int standardizeFlag = 0);
//	void standardizeInputTable(const shmea::GString&, int = 0, bool = true);
	void standardizeInputTable(const shmea::GTable&, int = 0, bool = true);

	virtual shmea::GVector<float> getTrainRow(unsigned int) const;
	virtual shmea::GVector<float> getTrainExpectedRow(unsigned int) const;

	virtual shmea::GVector<float> getTestRow(unsigned int) const;
	virtual shmea::GVector<float> getTestExpectedRow(unsigned int) const;

	// Zero-copy views into the underlying matrices.
	virtual bool getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;
	virtual bool getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;
	virtual bool getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;
	virtual bool getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const;

	// Sparse views (optional). When enabled and populated, these avoid dense materialization.
	virtual bool getTrainRowSparseView(unsigned int index,
	                                  const unsigned int*& outIndices,
	                                  const float*& outValues,
	                                  unsigned int& outNNZ,
	                                  unsigned int& outFullSize) const;
	virtual bool getTestRowSparseView(unsigned int index,
	                                 const unsigned int*& outIndices,
	                                 const float*& outValues,
	                                 unsigned int& outNNZ,
	                                 unsigned int& outFullSize) const;

	virtual unsigned int getTrainSize() const;
	virtual unsigned int getTestSize() const;
	virtual unsigned int getFeatureCount() const;

	// Fast shape contract (prevents TrainingCore from materializing rows to validate sizes).
	virtual bool hasFixedTrainRowSize() const { return (getFeatureCount() > 0u); }
	virtual unsigned int getFixedTrainRowSize() const { return getFeatureCount(); }
	virtual bool hasFixedTrainExpectedRowSize() const { return (expectedCountCached > 0u) || ((trainExpectedMatrix.size() > 0) && (trainExpectedMatrix[0].size() > 0)); }
	virtual unsigned int getFixedTrainExpectedRowSize() const
	{
		if (expectedCountCached > 0u)
			return expectedCountCached;
		if (trainExpectedMatrix.size() == 0)
			return 0u;
		return static_cast<unsigned int>(trainExpectedMatrix[0].size());
	}

	virtual int getType() const;

private:
	// Core encoder: fit OHE/scalers on TRAIN only, transform train + optional test.
	// This is the implementation behind importTrainTest() and importWithSplit().
	void standardizeInputTablesFitOnTrain(const shmea::GTable& trainRaw,
	                                      const shmea::GTable* testRaw,
	                                      int standardizeFlag,
	                                      bool changeValues);

	// Build packed row-major buffers from currently-populated dense matrices.
	// Returns true if packing was performed (or buffers are already valid).
	bool rebuildPackedDenseFromGMatrix();
};

};

#endif
