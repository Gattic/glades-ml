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
#include "NumberInput.h"
#include "Backend/Database/GTable.h"
#include "Backend/Database/SaveFolder.h"
#include "Backend/Database/SaveTable.h"
#include "../GMath/OHE.h"
#include "../GMath/gmath.h"
#include "tabular_preprocessing.h"
#include "MappedMatrix.h"
#include "../Structure/nninfo.h"
#include <vector>
#include <limits>
#include <map>
#include <string>
#include <cstdio> // snprintf
#include <sys/stat.h>
#ifdef _MSC_VER
#ifndef S_ISREG
#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#endif
#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif
#endif
#include <sys/stat.h>

using namespace glades;

namespace {
// Thread-local scratch for view APIs that must materialize sparse rows.
// We cannot return pointers to shared mutable scratch storage safely.
#if defined(__cplusplus) && (__cplusplus >= 201103L)
#define GLADES_THREAD_LOCAL thread_local
#elif defined(_MSC_VER)
#define GLADES_THREAD_LOCAL __declspec(thread)
#elif defined(__GNUC__) || defined(__clang__)
#define GLADES_THREAD_LOCAL __thread
#else
#define GLADES_THREAD_LOCAL
#endif

static inline bool ensure_tls_buf(unsigned int want, float*& buf, unsigned int& cap)
{
	if (want == 0u)
		return false;
	if (cap >= want && buf)
		return true;
	// best-effort resize
	delete[] buf;
	buf = NULL;
	cap = 0u;
	buf = new float[want];
	if (!buf)
		return false;
	cap = want;
	return true;
}

static inline bool safe_row_span(size_t flatSize, unsigned int row, unsigned int cols, size_t& outOff)
{
	outOff = 0u;
	if (cols == 0u)
		return false;
	const size_t c = static_cast<size_t>(cols);
	const size_t r = static_cast<size_t>(row);
	// overflow check: r*c must fit size_t
	if (c > 0u && r > (std::numeric_limits<size_t>::max() / c))
		return false;
	const size_t off = r * c;
	if (off > flatSize)
		return false;
	if ((flatSize - off) < c)
		return false;
	outOff = off;
	return true;
}

static inline std::string to_std_string(const shmea::GString& s)
{
    return std::string(s.c_str());
}

static inline bool path_is_file(const std::string& p)
{
    struct stat st;
    if (stat(p.c_str(), &st) != 0)
        return false;
    return S_ISREG(st.st_mode);
}

static inline bool path_is_dir(const std::string& p)
{
    struct stat st;
    if (stat(p.c_str(), &st) != 0)
        return false;
    return S_ISDIR(st.st_mode);
}

static inline bool ensure_dir_exists(const std::string& p)
{
	struct stat st;
	if (stat(p.c_str(), &st) == 0)
		return S_ISDIR(st.st_mode);
	// best-effort create
	return (mkdir(p.c_str(), 0755) == 0);
}

// Simple deterministic RNG (xorshift64*) for shuffling indices.
static inline uint64_t xorshift64s(uint64_t& state)
{
    if (state == 0u)
        state = 0x9e3779b97f4a7c15ULL;
    uint64_t x = state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state = x;
    return x * 2685821657736338717ULL;
}

static inline unsigned int urand(uint64_t& state, unsigned int bound)
{
    if (bound == 0u)
        return 0u;
    return static_cast<unsigned int>(xorshift64s(state) % static_cast<uint64_t>(bound));
}

static void shuffle_indices(std::vector<unsigned int>& idx, uint64_t seed)
{
    uint64_t s = seed;
    for (size_t i = idx.size(); i > 1; --i)
    {
        const unsigned int j = urand(s, static_cast<unsigned int>(i));
        const unsigned int k = static_cast<unsigned int>(i - 1);
        const unsigned int tmp = idx[k];
        idx[k] = idx[j];
        idx[j] = tmp;
    }
}

static bool find_single_output_col(const shmea::GTable& t, unsigned int& outCol)
{
    outCol = 0u;
    int count = 0;
    for (unsigned int c = 0; c < t.numberOfCols(); ++c)
    {
        if (t.isOutput(c))
        {
            outCol = c;
            ++count;
        }
    }
    return count == 1;
}

static std::string cell_to_string(const shmea::GType& cell)
{
    switch (cell.getType())
    {
    case shmea::GType::STRING_TYPE:
        return std::string(cell.c_str());
    case shmea::GType::CHAR_TYPE:
        return std::string(shmea::GString::intTOstring(cell.getChar()).c_str());
    case shmea::GType::SHORT_TYPE:
        return std::string(shmea::GString::intTOstring(cell.getShort()).c_str());
    case shmea::GType::INT_TYPE:
        return std::string(shmea::GString::intTOstring(cell.getInt()).c_str());
    case shmea::GType::LONG_TYPE:
        return std::string(shmea::GString::longTOstring(cell.getLong()).c_str());
    case shmea::GType::BOOLEAN_TYPE:
        return cell.getBoolean() ? "true" : "false";
    default:
        {
            // Defensive: avoid sprintf (buffer overflow risk). 64 bytes is plenty for %g, but be safe anyway.
            char buf[128];
            ::snprintf(buf, sizeof(buf), "%g", cell.getFloat());
            buf[sizeof(buf) - 1] = '\0';
            return std::string(buf);
        }
    }
}

static shmea::GTable table_from_indices(const shmea::GTable& src, const std::vector<unsigned int>& idx)
{
    shmea::GTable out(src.getDelimiter(), src.getHeaders());
    for (unsigned int c = 0; c < src.numberOfCols(); ++c)
        if (src.isOutput(c))
            out.toggleOutput(c);
    for (size_t i = 0; i < idx.size(); ++i)
        out.addRow(src.getRow(idx[i]));
    return out;
}
} // namespace

bool glades::NumberInput::rebuildPackedDenseFromGMatrix()
{
	// Only pack if enabled.
	if (!contiguousDenseEnabled)
		return false;

	// Pack trainX/trainY (required for "loaded" NumberInput in dense mode).
	// If matrices are empty, keep buffers cleared.
	trainXFlat.clear();
	trainYFlat.clear();
	testXFlat.clear();
	testYFlat.clear();
	trainRowsCachedDense = 0u;
	testRowsCachedDense = 0u;

	// Helper: flatten a GMatrix (vector-of-rows) into a single contiguous row-major vector.
	struct Flatten
	{
		static bool run(const shmea::GMatrix& m, std::vector<float>& out, unsigned int& outRows, unsigned int& outCols)
		{
			out.clear();
			outRows = 0u;
			outCols = 0u;
			const unsigned int R = static_cast<unsigned int>(m.size());
			if (R == 0u)
				return true;
			const unsigned int C = static_cast<unsigned int>(m[0].size());
			if (C == 0u)
				return true;
			// Validate non-ragged rows.
			for (unsigned int r = 0; r < R; ++r)
			{
				if (static_cast<unsigned int>(m[r].size()) != C)
					return false;
			}
			out.resize(static_cast<size_t>(R) * static_cast<size_t>(C));
			size_t k = 0;
			for (unsigned int r = 0; r < R; ++r)
			{
				const shmea::GVector<float>& row = m[r];
				for (unsigned int c = 0; c < C; ++c)
					out[k++] = row[c];
			}
			outRows = R;
			outCols = C;
			return true;
		}
	};

	unsigned int trR = 0u, trC = 0u;
	unsigned int trYR = 0u, trYC = 0u;
	if (!Flatten::run(trainMatrix, trainXFlat, trR, trC))
	{
		// Ragged matrices are a contract violation for training; do not build packed storage.
		trainXFlat.clear();
		return false;
	}
	if (!Flatten::run(trainExpectedMatrix, trainYFlat, trYR, trYC))
	{
		trainYFlat.clear();
		return false;
	}

	// Optional test split.
	unsigned int teR = 0u, teC = 0u;
	unsigned int teYR = 0u, teYC = 0u;
	if (!Flatten::run(testMatrix, testXFlat, teR, teC))
	{
		testXFlat.clear();
	}
	if (!Flatten::run(testExpectedMatrix, testYFlat, teYR, teYC))
	{
		testYFlat.clear();
	}

	trainRowsCachedDense = trR;
	testRowsCachedDense = teR;

	// Basic sanity: shapes should agree with cached counts when present.
	if (trR > 0u && trC > 0u)
		featureCountCached = trC;
	if (trYR > 0u && trYC > 0u)
		expectedCountCached = trYC;

	// If configured, drop the dense matrices to save RAM (the packed buffers become authoritative).
	if (!keepDenseMatrixInContiguousMode)
	{
		trainMatrix.clear();
		trainExpectedMatrix.clear();
		testMatrix.clear();
		testExpectedMatrix.clear();
	}

	return true;
}

// Fit encoding/scalers on TRAIN only and transform both train and optional test.
void glades::NumberInput::standardizeInputTablesFitOnTrain(const shmea::GTable& trainRaw,
                                                           const shmea::GTable* testRaw,
                                                           int standardizeFlag,
                                                           bool changeValues)
{
	glades::NumberInput& di = *this;
	if ((trainRaw.numberOfRows() <= 0) || (trainRaw.numberOfCols() <= 0))
		return;
	if (testRaw && (testRaw->numberOfCols() != trainRaw.numberOfCols()))
		return;

	// Fit preprocessing on TRAIN only (shared module).
	glades::TabularFitOptions fitOpt;
	// Infer categorical columns by scanning a bounded set of rows (more robust than first-row inference).
	fitOpt.inferMode = glades::TabularFitOptions::SCAN_ROWS_STRING;
	fitOpt.globalOutputOHEByCol = NULL;

	glades::TabularFit fit;
	std::string err;
	if (!glades::tabularFitOnTrain(trainRaw, fit, fitOpt, &err))
		return;

	// Cache logical shapes (used for sparse-only mode).
	di.featureCountCached = fit.totalInputDims;
	di.expectedCountCached = fit.totalOutputDims;

	// DataInput metadata for consumers.
	di.trainingOHEMaps = fit.oheByCol;
	di.testingOHEMaps = fit.oheByCol;
	di.trainingFeatureIsCategorical = fit.isCategorical;
	di.testingFeatureIsCategorical = fit.isCategorical;

	// Encode train + optional test.
	glades::TabularEncodeOptions encOpt;
	encOpt.standardizeFlag = standardizeFlag;
	encOpt.changeValues = changeValues;

	// Dense emission matches NumberInput behavior: sparse mode may keep dense matrices optionally.
	const bool wantDenseInputs = (!di.sparseInputEnabled) || di.keepDenseMatrixInSparseMode;
	encOpt.emitDenseInputs = wantDenseInputs;
	encOpt.emitDenseOutputs = true;
	encOpt.emitSparseInputs = di.sparseInputEnabled;

	glades::TabularEncoded enc;
	if (!glades::tabularTransformTrainTest(trainRaw, testRaw, fit, encOpt, enc, &err))
		return;

	di.trainMatrix = enc.trainX;
	di.trainExpectedMatrix = enc.trainY;
	di.testMatrix = enc.testX;
	di.testExpectedMatrix = enc.testY;
	di.trainSparseRows = enc.trainSparseX;
	di.testSparseRows = enc.testSparseX;

	di.min = fit.globalMin;
	di.max = fit.globalMax;

	// Optional packed dense storage for better cache locality in training hot paths.
	// This is an opt-in mode (see enableContiguousDense()).
	di.rebuildPackedDenseFromGMatrix();
}

void NumberInput::import(shmea::GString fname, int standardizeFlag)
{
    if(loaded)
    {
        return;
    }

    name = fname;

    // Support explicit train/test CSVs in a dataset directory:
    // - import("path/to/dataset_dir") loads "train.csv" + "test.csv" if present.
    // Backward compatibility: if fname is a file path, load that file as a single training table.
    const std::string p = to_std_string(fname);
    if (path_is_dir(p))
    {
        const std::string trainP = p + "/train.csv";
        const std::string testP = p + "/test.csv";
        if (path_is_file(trainP) && path_is_file(testP))
        {
            importTrainTest(shmea::GString(trainP.c_str()), shmea::GString(testP.c_str()), standardizeFlag);
            return;
        }
        // Fall through: directory exists but no train/test pair.
    }

    // Load and Normalize/Standardize the data (single-split).
    shmea::GTable rawTable = shmea::GTable(fname, ',', shmea::GTable::TYPE_FILE);
    standardizeInputTable(rawTable, standardizeFlag);

    // TODO: test table stuff

    // Set the loaded flag
    loaded = true;
}

void NumberInput::import(const shmea::GTable& rawTable, int standardizeFlag)
{
    if(loaded)
    {
        return;
    }

    // Load and Normalize/Standardize the data
    const bool changeValues = (standardizeFlag != GMath::NONE);
    standardizeInputTable(rawTable, standardizeFlag, changeValues);

    loaded = true;
}

bool NumberInput::importTrainTest(const shmea::GTable& trainRaw, const shmea::GTable& testRaw, int standardizeFlag)
{
    if (loaded)
        return true;

    const bool changeValues = (standardizeFlag != GMath::NONE);

    // Ensure test table uses the same output column configuration as train.
    // (Some call sites may have only set outputs on one table.)
    shmea::GTable testAdj = testRaw;
    for (unsigned int c = 0; c < trainRaw.numberOfCols(); ++c)
    {
        const bool wantOut = trainRaw.isOutput(c);
        const bool hasOut = testAdj.isOutput(c);
        if (wantOut != hasOut)
            testAdj.toggleOutput(c);
    }

    standardizeInputTablesFitOnTrain(trainRaw, &testAdj, standardizeFlag, changeValues);
    loaded = true;
    return true;
}

bool NumberInput::importTrainTest(shmea::GString trainFile, shmea::GString testFile, int standardizeFlag)
{
    if (loaded)
        return true;
    shmea::GTable trainRaw = shmea::GTable(trainFile, ',', shmea::GTable::TYPE_FILE);
    shmea::GTable testRaw = shmea::GTable(testFile, ',', shmea::GTable::TYPE_FILE);
    // NOTE: output columns are unknown at this point; this overload is mainly a convenience for
    // users whose datasets already encode output columns in the GTable import pipeline.
    return importTrainTest(trainRaw, testRaw, standardizeFlag);
}

bool NumberInput::importWithSplit(const shmea::GTable& rawTable, const TrainTestSplitConfig& cfg, int standardizeFlag)
{
    if (loaded)
        return true;

    const unsigned int N = rawTable.numberOfRows();
    if (N == 0u || rawTable.numberOfCols() == 0u)
        return false;

    float frac = cfg.testFraction;
    if (frac < 0.0f) frac = 0.0f;
    if (frac > 1.0f) frac = 1.0f;

    unsigned int wantTest = static_cast<unsigned int>(static_cast<double>(N) * static_cast<double>(frac) + 0.5);
    if (frac > 0.0f && wantTest == 0u) wantTest = 1u;
    if (wantTest >= N && N > 1u) wantTest = N - 1u; // keep at least 1 train row

    std::vector<unsigned int> trainIdx;
    std::vector<unsigned int> testIdx;
    trainIdx.reserve(N);
    testIdx.reserve(wantTest);

    // Best-effort stratify by a single output column.
    unsigned int outCol = 0u;
    const bool canStratify = cfg.stratify && find_single_output_col(rawTable, outCol);
    if (canStratify)
    {
        std::map<std::string, std::vector<unsigned int> > byLabel;
        for (unsigned int i = 0; i < N; ++i)
            byLabel[cell_to_string(rawTable.getCell(i, outCol))].push_back(i);

        // Shuffle within each group deterministically (if enabled).
        if (cfg.shuffle)
        {
            for (std::map<std::string, std::vector<unsigned int> >::iterator it = byLabel.begin(); it != byLabel.end(); ++it)
            {
                // derive per-label seed (djb2)
                uint64_t h = 5381u;
                for (size_t j = 0; j < it->first.size(); ++j)
                    h = ((h << 5) + h) + static_cast<uint64_t>(static_cast<unsigned char>(it->first[j]));
                shuffle_indices(it->second, cfg.seed ^ (h + 0x9e3779b97f4a7c15ULL));
            }
        }

        // Round-robin fill test set while ensuring each label keeps at least 1 train sample (when possible).
        while (testIdx.size() < wantTest)
        {
            bool progressed = false;
            for (std::map<std::string, std::vector<unsigned int> >::iterator it = byLabel.begin(); it != byLabel.end(); ++it)
            {
                if (testIdx.size() >= wantTest)
                    break;
                // Leave at least one in train if group has >1.
                if (it->second.size() > 1u)
                {
                    testIdx.push_back(it->second.back());
                    it->second.pop_back();
                    progressed = true;
                }
            }
            if (!progressed)
                break; // cannot take more without emptying a group
        }

        // Remaining indices => train (including any leftover that couldn't be assigned to test).
        for (std::map<std::string, std::vector<unsigned int> >::iterator it = byLabel.begin(); it != byLabel.end(); ++it)
            for (size_t j = 0; j < it->second.size(); ++j)
                trainIdx.push_back(it->second[j]);
    }
    else
    {
        std::vector<unsigned int> idx;
        idx.reserve(N);
        for (unsigned int i = 0; i < N; ++i)
            idx.push_back(i);
        if (cfg.shuffle)
            shuffle_indices(idx, cfg.seed);
        const unsigned int split = N - wantTest;
        for (unsigned int i = 0; i < split; ++i)
            trainIdx.push_back(idx[i]);
        for (unsigned int i = split; i < N; ++i)
            testIdx.push_back(idx[i]);
    }

    if (trainIdx.empty())
        return false;

    const shmea::GTable trainTbl = table_from_indices(rawTable, trainIdx);
    const shmea::GTable testTbl = testIdx.empty() ? shmea::GTable(rawTable.getDelimiter(), rawTable.getHeaders())
                                                  : table_from_indices(rawTable, testIdx);
    // Ensure outputs match rawTable.
    shmea::GTable testAdj = testTbl;
    for (unsigned int c = 0; c < rawTable.numberOfCols(); ++c)
    {
        const bool wantOut = rawTable.isOutput(c);
        const bool hasOut = testAdj.isOutput(c);
        if (wantOut != hasOut)
            testAdj.toggleOutput(c);
    }

    const bool changeValues = (standardizeFlag != GMath::NONE);
    standardizeInputTablesFitOnTrain(trainTbl, testIdx.empty() ? NULL : &testAdj, standardizeFlag, changeValues);
    loaded = true;
    return true;
}

//void glades::NumberInput::standardizeInputTable(const shmea::GString& inputFName, int standardizeFlag, bool changeValues)
void glades::NumberInput::standardizeInputTable(const shmea::GTable& rawTable, int standardizeFlag, bool changeValues)
{
    // Backward-compatible single-split path: fit on this table and transform train only.
    standardizeInputTablesFitOnTrain(rawTable, NULL, standardizeFlag, changeValues);
}

shmea::GVector<float> NumberInput::getTrainRow(unsigned int index) const
{
	// Prefer dense storage if available.
	if (index < trainMatrix.size())
		return trainMatrix[index];

	// Packed dense storage path (row-major contiguous).
	if (contiguousDenseEnabled && !trainXFlat.empty() && featureCountCached > 0u && index < trainRowsCachedDense)
	{
		// Return by value: do not use shared mutable scratch in a const method.
		size_t off = 0u;
		if (!safe_row_span(trainXFlat.size(), index, featureCountCached, off))
			return emptyRow;
		shmea::GVector<float> row(featureCountCached, 0.0f);
		const float* p = &trainXFlat[off];
		for (unsigned int i = 0; i < featureCountCached; ++i)
			row[i] = p[i];
		return row;
	}

	// Sparse-only mode: materialize without returning shared scratch.
	if (sparseInputEnabled && index < trainSparseRows.size() && featureCountCached > 0u)
	{
		shmea::GVector<float> row(featureCountCached, 0.0f);
		const SparseRow& sr = trainSparseRows[index];
		const unsigned int nnz = static_cast<unsigned int>(sr.idx.size());
		for (unsigned int k = 0; k < nnz; ++k)
		{
			const unsigned int fi = sr.idx[k];
			if (fi < featureCountCached)
				row[fi] = sr.val[k];
		}
		return row;
	}

	return emptyRow;
}

shmea::GVector<float> NumberInput::getTrainExpectedRow(unsigned int index) const
{
	if (index < trainExpectedMatrix.size())
		return trainExpectedMatrix[index];

	if (contiguousDenseEnabled && !trainYFlat.empty() && expectedCountCached > 0u && index < trainRowsCachedDense)
	{
		size_t off = 0u;
		if (!safe_row_span(trainYFlat.size(), index, expectedCountCached, off))
			return emptyRow;
		shmea::GVector<float> row(expectedCountCached, 0.0f);
		const float* p = &trainYFlat[off];
		for (unsigned int i = 0; i < expectedCountCached; ++i)
			row[i] = p[i];
		return row;
	}

	return emptyRow;
}

shmea::GVector<float> NumberInput::getTestRow(unsigned int index) const
{
	if (index < testMatrix.size())
		return testMatrix[index];

	if (contiguousDenseEnabled && !testXFlat.empty() && featureCountCached > 0u && index < testRowsCachedDense)
	{
		size_t off = 0u;
		if (!safe_row_span(testXFlat.size(), index, featureCountCached, off))
			return shmea::GVector<float>();
		shmea::GVector<float> row(featureCountCached, 0.0f);
		const float* p = &testXFlat[off];
		for (unsigned int i = 0; i < featureCountCached; ++i)
			row[i] = p[i];
		return row;
	}

	if (sparseInputEnabled && index < testSparseRows.size() && featureCountCached > 0u)
	{
		shmea::GVector<float> row(featureCountCached, 0.0f);
		const SparseRow& sr = testSparseRows[index];
		const unsigned int nnz = static_cast<unsigned int>(sr.idx.size());
		for (unsigned int k = 0; k < nnz; ++k)
		{
			const unsigned int fi = sr.idx[k];
			if (fi < featureCountCached)
				row[fi] = sr.val[k];
		}
		return row;
	}

	return shmea::GVector<float>();
}

shmea::GVector<float> NumberInput::getTestExpectedRow(unsigned int index) const
{
	if (index < testExpectedMatrix.size())
		return testExpectedMatrix[index];

	if (contiguousDenseEnabled && !testYFlat.empty() && expectedCountCached > 0u && index < testRowsCachedDense)
	{
		size_t off = 0u;
		if (!safe_row_span(testYFlat.size(), index, expectedCountCached, off))
			return shmea::GVector<float>();
		shmea::GVector<float> row(expectedCountCached, 0.0f);
		const float* p = &testYFlat[off];
		for (unsigned int i = 0; i < expectedCountCached; ++i)
			row[i] = p[i];
		return row;
	}

	return shmea::GVector<float>();
}

bool NumberInput::getTrainRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;

	// Packed dense fast path.
	if (contiguousDenseEnabled && !trainXFlat.empty() && featureCountCached > 0u && index < trainRowsCachedDense)
	{
		outData = &trainXFlat[static_cast<size_t>(index) * static_cast<size_t>(featureCountCached)];
		outSize = featureCountCached;
		return true;
	}

	// Dense fast path.
	if (index < trainMatrix.size())
	{
		const shmea::GVector<float>& row = trainMatrix[index];
		if (row.size() == 0)
			return false;
		outData = row.data();
		outSize = static_cast<unsigned int>(row.size());
		return (outData != NULL);
	}

	// Sparse-only path: materialize into scratch dense storage.
	if (sparseInputEnabled && index < trainSparseRows.size() && featureCountCached > 0u)
	{
		static GLADES_THREAD_LOCAL float* tlsBuf = NULL;
		static GLADES_THREAD_LOCAL unsigned int tlsCap = 0u;
		if (!ensure_tls_buf(featureCountCached, tlsBuf, tlsCap))
			return false;
		for (unsigned int i = 0; i < featureCountCached; ++i)
			tlsBuf[i] = 0.0f;
		const SparseRow& sr = trainSparseRows[index];
		const unsigned int nnz = static_cast<unsigned int>(sr.idx.size());
		for (unsigned int k = 0; k < nnz; ++k)
		{
			const unsigned int fi = sr.idx[k];
			if (fi < featureCountCached)
				tlsBuf[fi] = sr.val[k];
		}
		outData = tlsBuf;
		outSize = featureCountCached;
		return true;
	}

	return false;
}

bool NumberInput::getTrainExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;

	// Packed dense fast path.
	if (contiguousDenseEnabled && !trainYFlat.empty() && expectedCountCached > 0u && index < trainRowsCachedDense)
	{
		outData = &trainYFlat[static_cast<size_t>(index) * static_cast<size_t>(expectedCountCached)];
		outSize = expectedCountCached;
		return true;
	}

	if (index >= trainExpectedMatrix.size())
		return false;
	const shmea::GVector<float>& row = trainExpectedMatrix[index];
	if (row.size() == 0)
		return false;
	outData = row.data();
	outSize = static_cast<unsigned int>(row.size());
	return (outData != NULL);
}

bool NumberInput::getTestRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;

	// Packed dense fast path.
	if (contiguousDenseEnabled && !testXFlat.empty() && featureCountCached > 0u && index < testRowsCachedDense)
	{
		outData = &testXFlat[static_cast<size_t>(index) * static_cast<size_t>(featureCountCached)];
		outSize = featureCountCached;
		return true;
	}

	if (index < testMatrix.size())
	{
		const shmea::GVector<float>& row = testMatrix[index];
		if (row.size() == 0)
			return false;
		outData = row.data();
		outSize = static_cast<unsigned int>(row.size());
		return (outData != NULL);
	}

	if (sparseInputEnabled && index < testSparseRows.size() && featureCountCached > 0u)
	{
		static GLADES_THREAD_LOCAL float* tlsBuf = NULL;
		static GLADES_THREAD_LOCAL unsigned int tlsCap = 0u;
		if (!ensure_tls_buf(featureCountCached, tlsBuf, tlsCap))
			return false;
		for (unsigned int i = 0; i < featureCountCached; ++i)
			tlsBuf[i] = 0.0f;
		const SparseRow& sr = testSparseRows[index];
		const unsigned int nnz = static_cast<unsigned int>(sr.idx.size());
		for (unsigned int k = 0; k < nnz; ++k)
		{
			const unsigned int fi = sr.idx[k];
			if (fi < featureCountCached)
				tlsBuf[fi] = sr.val[k];
		}
		outData = tlsBuf;
		outSize = featureCountCached;
		return true;
	}

	return false;
}

bool NumberInput::getTrainRowSparseView(unsigned int index,
                                       const unsigned int*& outIndices,
                                       const float*& outValues,
                                       unsigned int& outNNZ,
                                       unsigned int& outFullSize) const
{
	outIndices = NULL;
	outValues = NULL;
	outNNZ = 0u;
	outFullSize = 0u;

	if (!sparseInputEnabled)
		return false;
	if (index >= trainSparseRows.size())
		return false;
	if (featureCountCached == 0u)
		return false;

	const SparseRow& sr = trainSparseRows[index];
	if (sr.idx.size() != sr.val.size())
		return false;

	outIndices = (sr.idx.empty() ? NULL : &sr.idx[0]);
	outValues = (sr.val.empty() ? NULL : &sr.val[0]);
	outNNZ = static_cast<unsigned int>(sr.idx.size());
	outFullSize = featureCountCached;
	return true;
}

bool NumberInput::getTestRowSparseView(unsigned int index,
                                      const unsigned int*& outIndices,
                                      const float*& outValues,
                                      unsigned int& outNNZ,
                                      unsigned int& outFullSize) const
{
	outIndices = NULL;
	outValues = NULL;
	outNNZ = 0u;
	outFullSize = 0u;

	if (!sparseInputEnabled)
		return false;
	if (index >= testSparseRows.size())
		return false;
	if (featureCountCached == 0u)
		return false;

	const SparseRow& sr = testSparseRows[index];
	if (sr.idx.size() != sr.val.size())
		return false;

	outIndices = (sr.idx.empty() ? NULL : &sr.idx[0]);
	outValues = (sr.val.empty() ? NULL : &sr.val[0]);
	outNNZ = static_cast<unsigned int>(sr.idx.size());
	outFullSize = featureCountCached;
	return true;
}

bool NumberInput::getTestExpectedRowView(unsigned int index, const float*& outData, unsigned int& outSize) const
{
	outData = NULL;
	outSize = 0u;

	// Packed dense fast path.
	if (contiguousDenseEnabled && !testYFlat.empty() && expectedCountCached > 0u && index < testRowsCachedDense)
	{
		outData = &testYFlat[static_cast<size_t>(index) * static_cast<size_t>(expectedCountCached)];
		outSize = expectedCountCached;
		return true;
	}

	if (index >= testExpectedMatrix.size())
		return false;
	const shmea::GVector<float>& row = testExpectedMatrix[index];
	if (row.size() == 0)
		return false;
	outData = row.data();
	outSize = static_cast<unsigned int>(row.size());
	return (outData != NULL);
}

unsigned int NumberInput::getTrainSize() const
{
	if (trainMatrix.size() > 0)
		return trainMatrix.size();
	if (contiguousDenseEnabled && trainRowsCachedDense > 0u)
		return trainRowsCachedDense;
	return static_cast<unsigned int>(trainSparseRows.size());
}

unsigned int NumberInput::getTestSize() const
{
	if (testMatrix.size() > 0)
		return testMatrix.size();
	if (contiguousDenseEnabled && testRowsCachedDense > 0u)
		return testRowsCachedDense;
	return static_cast<unsigned int>(testSparseRows.size());
}

unsigned int NumberInput::getFeatureCount() const
{
	if (trainMatrix.size() > 0)
		return trainMatrix[0].size();
	return featureCountCached;
}

int NumberInput::getType() const
{
    return CSV;
}

bool glades::NumberInput::exportMappedDataset(const std::string& dirPath, std::string* errMsg) const
{
	if (!loaded)
	{
		if (errMsg) *errMsg = "exportMappedDataset: NumberInput is not loaded";
		return false;
	}
	// Prefer dense matrices, but allow exporting from packed dense buffers too.
	const bool haveTrainDense = (trainMatrix.size() > 0 && trainExpectedMatrix.size() > 0);
	const bool haveTrainPacked =
	    (contiguousDenseEnabled && !trainXFlat.empty() && !trainYFlat.empty() && trainRowsCachedDense > 0u &&
	     featureCountCached > 0u && expectedCountCached > 0u);
	if (!haveTrainDense && !haveTrainPacked)
	{
		if (errMsg) *errMsg = "exportMappedDataset: train matrices are empty";
		return false;
	}
	if (haveTrainDense && (trainMatrix.size() != trainExpectedMatrix.size()))
	{
		if (errMsg) *errMsg = "exportMappedDataset: trainMatrix rows != trainExpectedMatrix rows";
		return false;
	}
	if (!ensure_dir_exists(dirPath))
	{
		if (errMsg) *errMsg = "exportMappedDataset: failed to create directory (or path is not a directory)";
		return false;
	}

	std::string werr;
	if (haveTrainDense)
	{
		if (!glades::MappedFloatMatrix::writeFromGMatrix(dirPath + "/train.x.gcol", trainMatrix, &werr))
		{
			if (errMsg) *errMsg = std::string("exportMappedDataset: write train.x.gcol failed: ") + werr;
			return false;
		}
		if (!glades::MappedFloatMatrix::writeFromGMatrix(dirPath + "/train.y.gcol", trainExpectedMatrix, &werr))
		{
			if (errMsg) *errMsg = std::string("exportMappedDataset: write train.y.gcol failed: ") + werr;
			return false;
		}
	}
	else
	{
		// Packed dense (row-major).
		if (!glades::MappedFloatMatrix::writeFromDense(dirPath + "/train.x.gcol",
		                                               static_cast<unsigned long long>(trainRowsCachedDense),
		                                               static_cast<unsigned long long>(featureCountCached),
		                                               &trainXFlat[0],
		                                               &werr))
		{
			if (errMsg) *errMsg = std::string("exportMappedDataset: write train.x.gcol failed: ") + werr;
			return false;
		}
		if (!glades::MappedFloatMatrix::writeFromDense(dirPath + "/train.y.gcol",
		                                               static_cast<unsigned long long>(trainRowsCachedDense),
		                                               static_cast<unsigned long long>(expectedCountCached),
		                                               &trainYFlat[0],
		                                               &werr))
		{
			if (errMsg) *errMsg = std::string("exportMappedDataset: write train.y.gcol failed: ") + werr;
			return false;
		}
	}

	// Optional test split.
	const bool haveTestDense = (testMatrix.size() > 0 && testExpectedMatrix.size() > 0);
	const bool haveTestPacked =
	    (contiguousDenseEnabled && !testXFlat.empty() && !testYFlat.empty() && testRowsCachedDense > 0u &&
	     featureCountCached > 0u && expectedCountCached > 0u);
	if (haveTestDense || haveTestPacked || (testMatrix.size() > 0 || testExpectedMatrix.size() > 0))
	{
		if (!haveTestDense && !haveTestPacked)
		{
			if (errMsg) *errMsg = "exportMappedDataset: test split is partially populated";
			return false;
		}
		if (haveTestDense && (testMatrix.size() != testExpectedMatrix.size()))
		{
			if (errMsg) *errMsg = "exportMappedDataset: testMatrix rows != testExpectedMatrix rows";
			return false;
		}
		if (haveTestDense)
		{
			if (!glades::MappedFloatMatrix::writeFromGMatrix(dirPath + "/test.x.gcol", testMatrix, &werr))
			{
				if (errMsg) *errMsg = std::string("exportMappedDataset: write test.x.gcol failed: ") + werr;
				return false;
			}
			if (!glades::MappedFloatMatrix::writeFromGMatrix(dirPath + "/test.y.gcol", testExpectedMatrix, &werr))
			{
				if (errMsg) *errMsg = std::string("exportMappedDataset: write test.y.gcol failed: ") + werr;
				return false;
			}
		}
		else
		{
			if (!glades::MappedFloatMatrix::writeFromDense(dirPath + "/test.x.gcol",
			                                               static_cast<unsigned long long>(testRowsCachedDense),
			                                               static_cast<unsigned long long>(featureCountCached),
			                                               &testXFlat[0],
			                                               &werr))
			{
				if (errMsg) *errMsg = std::string("exportMappedDataset: write test.x.gcol failed: ") + werr;
				return false;
			}
			if (!glades::MappedFloatMatrix::writeFromDense(dirPath + "/test.y.gcol",
			                                               static_cast<unsigned long long>(testRowsCachedDense),
			                                               static_cast<unsigned long long>(expectedCountCached),
			                                               &testYFlat[0],
			                                               &werr))
			{
				if (errMsg) *errMsg = std::string("exportMappedDataset: write test.y.gcol failed: ") + werr;
				return false;
			}
		}
	}

	return true;
}
