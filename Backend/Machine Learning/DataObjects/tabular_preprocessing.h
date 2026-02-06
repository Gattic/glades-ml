#ifndef GLADES_TABULAR_PREPROCESSING_H
#define GLADES_TABULAR_PREPROCESSING_H
// Centralized tabular preprocessing (fit on train, transform train/test).
//
// This module is intentionally shared by:
// - NumberInput (CSV/tabular import)
// - cross_validation.cpp (k-fold / train-test evaluation on GTable)
//
// Goals:
// - Single source of truth for: column typing, OHE fitting, numeric scaler fitting, and dense/sparse encoding.
//
// NOTE: This codebase targets C++98 (no unordered_map, no auto, no nullptr).

#include "Backend/Database/GPointer.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/GVector.h"
#include "../GMath/OHE.h"

#include <string>
#include <vector>

namespace glades {

// Sparse feature row: nnz (index,value) pairs over an implicit dense vector.
struct TabularSparseRow
{
	std::vector<unsigned int> idx;
	std::vector<float> val;
};

struct TabularFit
{
	struct NumericStats
	{
		// Only valid when the column is numeric (isCategorical=false).
		float minv;
		float maxv;
		double mean;
		float stdev;
		// Missingness tracking (fit-time): counts of finite vs non-finite (NaN/Inf) values.
		// These allow callers to surface data-quality diagnostics without changing the encoded feature space.
		unsigned long long finiteCount;
		unsigned long long missingCount;
		NumericStats()
		    : minv(0.0f),
		      maxv(0.0f),
		      mean(0.0),
		      stdev(0.0f),
		      finiteCount(0ULL),
		      missingCount(0ULL)
		{
		}
	};

	unsigned int cols;
	std::vector<bool> isOutput;       // size=cols
	std::vector<bool> isCategorical;  // size=cols
	std::vector< shmea::GPointer<glades::OHE> > oheByCol; // size=cols (empty OHE for numeric columns)
	std::vector<NumericStats> numeric;                    // size=cols

	// Column dimensions after encoding (categorical -> OHE size, numeric -> 1).
	std::vector<unsigned int> colDim; // size=cols

	// Offsets into the encoded dense vectors (input/output feature spaces).
	std::vector<unsigned int> inputOffset;  // size=cols (meaningful when !isOutput[c])
	std::vector<unsigned int> outputOffset; // size=cols (meaningful when isOutput[c])

	unsigned int totalInputDims;
	unsigned int totalOutputDims;

	// Global numeric min/max across all numeric columns (train only).
	float globalMin;
	float globalMax;
	bool sawNumeric;

	TabularFit()
	    : cols(0u),
	      isOutput(),
	      isCategorical(),
	      oheByCol(),
	      numeric(),
	      colDim(),
	      inputOffset(),
	      outputOffset(),
	      totalInputDims(0u),
	      totalOutputDims(0u),
	      globalMin(0.0f),
	      globalMax(0.0f),
	      sawNumeric(false)
	{
	}
};

struct TabularFitOptions
{
	// How to infer categorical columns. Current engine behavior is FIRST_ROW_STRING.
	enum InferMode
	{
		FIRST_ROW_STRING = 0,
		SCAN_ROWS_STRING = 1
	};

	InferMode inferMode;
	unsigned int scanRows; // used only when inferMode==SCAN_ROWS_STRING

	// Optional pre-fitted output OHE maps (by original column index).
	// If provided, categorical output columns will reuse these to keep output dimensions stable
	// across folds/splits even when a class is missing from the train subset.
	const std::vector< shmea::GPointer<glades::OHE> >* globalOutputOHEByCol;

	TabularFitOptions()
	    : inferMode(FIRST_ROW_STRING),
	      scanRows(64u),
	      globalOutputOHEByCol(NULL)
	{
	}
};

struct TabularEncodeOptions
{
	// Numeric scaling.
	int standardizeFlag; // glades::GMath::* (NONE/MINMAX/ZSCORE)
	bool changeValues;   // if false, numeric values are not scaled

	// Dense encoding emission.
	bool emitDenseInputs;
	bool emitDenseOutputs;

	// Sparse input encoding emission (high-cardinality categorical inputs).
	// Sparse encoding always uses strict one-hot semantics for categorical *inputs* (1/0).
	bool emitSparseInputs;

	TabularEncodeOptions()
	    : standardizeFlag(0),
	      changeValues(true),
	      emitDenseInputs(true),
	      emitDenseOutputs(true),
	      emitSparseInputs(false)
	{
	}
};

struct TabularEncoded
{
	shmea::GMatrix trainX;
	shmea::GMatrix trainY;
	shmea::GMatrix testX;
	shmea::GMatrix testY;

	std::vector<TabularSparseRow> trainSparseX;
	std::vector<TabularSparseRow> testSparseX;
};

// Fit preprocessing on train only.
// Returns false on invalid input; errMsg (optional) describes the reason.
bool tabularFitOnTrain(const shmea::GTable& train,
                       TabularFit& outFit,
                       const TabularFitOptions& opt,
                       std::string* errMsg);

// Transform train + optional test using a pre-fit. Produces dense matrices and optionally sparse input rows.
bool tabularTransformTrainTest(const shmea::GTable& train,
                               const shmea::GTable* test,
                               const TabularFit& fit,
                               const TabularEncodeOptions& encOpt,
                               TabularEncoded& out,
                               std::string* errMsg);

} // namespace glades

#endif

