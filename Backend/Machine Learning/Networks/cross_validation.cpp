// Copyright 2026 Robert Carneiro, Derek Meer, Matthew Tabak, Eric Lujan
//
// Cross-validation runner (modern, deterministic).

#include "cross_validation.h"

#include "network.h"
#include "../DataObjects/NumberInput.h"
#include "../DataObjects/tabular_preprocessing.h"
#include "../GMath/gmath.h"
#include "../GMath/OHE.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdio.h>
#include <string>

using namespace glades;

namespace {

// Simple deterministic RNG (xorshift64*) suitable for shuffling indices.
static inline uint64_t xorshift64s(uint64_t& state)
{
	// Avoid a zero state.
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
	// Modulo bias is fine for shuffling indices at this scale.
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

static bool is_string_column(const shmea::GTable& t, unsigned int col)
{
	if (t.numberOfRows() == 0 || col >= t.numberOfCols())
		return false;
	return t.getCell(0u, col).getType() == shmea::GType::STRING_TYPE;
}

// Choose a single output column for stratification. Best-effort:
// - If exactly one output column exists, return it.
// - Otherwise return false.
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
	// Match ImageInput's label normalization approach: interpret integer/boolean types as strings too.
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
		// FLOAT/DOUBLE and others are treated as numeric; stringify for stratify grouping.
		{
			char buf[64];
			sprintf(buf, "%g", cell.getFloat());
			return std::string(buf);
		}
	}
}

struct FoldSplit
{
	std::vector<unsigned int> trainIdx;
	std::vector<unsigned int> testIdx;
};

static void build_kfold_splits(const shmea::GTable& t,
                               unsigned int k,
                               bool shuffle,
                               uint64_t seed,
                               bool stratify,
                               std::vector<FoldSplit>& out)
{
	out.clear();
	if (k == 0u)
		return;

	const unsigned int N = t.numberOfRows();
	if (N == 0u)
		return;

	// Clamp folds to at most N.
	if (k > N)
		k = N;

	// Base index pool.
	std::vector<unsigned int> idx;
	idx.reserve(N);
	for (unsigned int i = 0; i < N; ++i)
		idx.push_back(i);

	// Optional stratification.
	unsigned int outCol = 0u;
	const bool canStratify = stratify && find_single_output_col(t, outCol);
	if (canStratify)
	{
		// Group indices by label.
		std::map<std::string, std::vector<unsigned int> > byLabel;
		for (unsigned int i = 0; i < N; ++i)
		{
			const std::string lab = cell_to_string(t.getCell(i, outCol));
			byLabel[lab].push_back(i);
		}

		// Shuffle within each label group (if enabled).
		if (shuffle)
		{
			uint64_t s = seed;
			for (std::map<std::string, std::vector<unsigned int> >::iterator it = byLabel.begin(); it != byLabel.end(); ++it)
			{
				// Derive a per-label seed deterministically.
				// Hash: djb2
				uint64_t h = 5381u;
				for (size_t i = 0; i < it->first.size(); ++i)
					h = ((h << 5) + h) + static_cast<uint64_t>(static_cast<unsigned char>(it->first[i]));
				uint64_t labelSeed = s ^ (h + 0x9e3779b97f4a7c15ULL);
				shuffle_indices(it->second, labelSeed);
			}
		}

		// Round-robin assign to folds.
		out.resize(k);
		unsigned int f = 0u;
		for (std::map<std::string, std::vector<unsigned int> >::iterator it = byLabel.begin(); it != byLabel.end(); ++it)
		{
			for (size_t j = 0; j < it->second.size(); ++j)
			{
				out[f].testIdx.push_back(it->second[j]);
				f = (f + 1u) % k;
			}
		}

		// Train = complement.
		std::vector<char> isTest(N, 0);
		for (unsigned int fi = 0; fi < k; ++fi)
		{
			std::fill(isTest.begin(), isTest.end(), 0);
			for (size_t j = 0; j < out[fi].testIdx.size(); ++j)
				isTest[out[fi].testIdx[j]] = 1;
			out[fi].trainIdx.clear();
			out[fi].trainIdx.reserve(N - out[fi].testIdx.size());
			for (unsigned int i = 0; i < N; ++i)
				if (!isTest[i])
					out[fi].trainIdx.push_back(i);
		}
		return;
	}

	// Non-stratified shuffle.
	if (shuffle)
		shuffle_indices(idx, seed);

	// Split into k contiguous chunks as test folds.
	out.resize(k);
	for (unsigned int fi = 0; fi < k; ++fi)
	{
		const unsigned int start = (fi * N) / k;
		const unsigned int end = ((fi + 1u) * N) / k;
		out[fi].testIdx.assign(idx.begin() + start, idx.begin() + end);
		// Train = everything else.
		out[fi].trainIdx.clear();
		out[fi].trainIdx.reserve(N - (end - start));
		for (unsigned int j = 0; j < start; ++j)
			out[fi].trainIdx.push_back(idx[j]);
		for (unsigned int j = end; j < N; ++j)
			out[fi].trainIdx.push_back(idx[j]);
	}
}

static void build_walk_forward_splits(unsigned int N, unsigned int k, std::vector<FoldSplit>& out)
{
	out.clear();
	if (N == 0u || k == 0u)
		return;
	if (k > N)
		k = N;

	out.reserve(k);
	for (unsigned int fi = 0; fi < k; ++fi)
	{
		const unsigned int start = (fi * N) / k;
		const unsigned int end = ((fi + 1u) * N) / k;

		FoldSplit s;
		// Train = all rows before the test window.
		s.trainIdx.reserve(start);
		for (unsigned int i = 0; i < start; ++i)
			s.trainIdx.push_back(i);

		// Test = the window [start, end).
		s.testIdx.reserve(end - start);
		for (unsigned int i = start; i < end; ++i)
			s.testIdx.push_back(i);

		out.push_back(s);
	}
}

static shmea::GTable table_from_indices(const shmea::GTable& src, const std::vector<unsigned int>& idx)
{
	shmea::GTable out(src.getDelimiter(), src.getHeaders());
	// Preserve which columns are outputs.
	for (unsigned int c = 0; c < src.numberOfCols(); ++c)
		if (src.isOutput(c))
			out.toggleOutput(c);
	for (size_t i = 0; i < idx.size(); ++i)
		out.addRow(src.getRow(idx[i]));
	return out;
}

static void fit_output_label_ohe_on_train_only(const shmea::GTable& trainTbl,
                                              std::vector< shmea::GPointer<glades::OHE> >& outOheByCol)
{
	outOheByCol.clear();
	outOheByCol.resize(trainTbl.numberOfCols());
	for (unsigned int c = 0; c < trainTbl.numberOfCols(); ++c)
	{
		if (trainTbl.isOutput(c) && is_string_column(trainTbl, c))
		{
			shmea::GPointer<glades::OHE> o(new glades::OHE());
			o->mapFeatureSpace(trainTbl, static_cast<int>(c));
			outOheByCol[c] = o;
		}
	}
}

static glades::NNetworkStatus run_one_fold(const shmea::GTable& trainTbl,
                                          const shmea::GTable& testTbl,
                                          const std::vector<glades::NNetwork*>& templates,
                                          const CrossValidationConfig& cfg,
                                          const std::vector< shmea::GPointer<glades::OHE> >& globalOutputOHEByCol,
                                          std::vector<float>& outFoldAcc)
{
	outFoldAcc.assign(templates.size(), 0.0f);
	if (templates.empty())
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "crossValidate: modelTemplates is empty");

	// Fit/transform for CSV into NumberInput with both train/test splits populated.
	glades::TabularFitOptions fitOpt;
	// Infer categorical columns by scanning a bounded set of rows (more robust than first-row inference).
	fitOpt.inferMode = glades::TabularFitOptions::SCAN_ROWS_STRING;
	fitOpt.globalOutputOHEByCol = &globalOutputOHEByCol;

	glades::TabularFit fit;
	std::string err;
	if (!glades::tabularFitOnTrain(trainTbl, fit, fitOpt, &err))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "crossValidate: failed to fit preprocessing on fold train table");

	if (fit.totalInputDims == 0u || fit.totalOutputDims == 0u)
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "crossValidate: invalid input/output dimensionality from table");

	glades::TabularEncodeOptions encOpt;
	encOpt.standardizeFlag = cfg.standardizeFlag;
	encOpt.changeValues = (cfg.standardizeFlag != glades::GMath::NONE);
	encOpt.emitDenseInputs = true;
	encOpt.emitDenseOutputs = true;
	encOpt.emitSparseInputs = false;

	glades::TabularEncoded enc;
	if (!glades::tabularTransformTrainTest(trainTbl, &testTbl, fit, encOpt, enc, &err))
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "crossValidate: failed to transform fold tables");

	if (enc.trainX.size() == 0 || enc.testX.size() == 0)
		return glades::NNetworkStatus(glades::NNetworkStatus::EMPTY_DATA, "crossValidate: empty fold after splitting");

	glades::NumberInput di;
	di.trainMatrix = enc.trainX;
	di.trainExpectedMatrix = enc.trainY;
	di.testMatrix = enc.testX;
	di.testExpectedMatrix = enc.testY;
	di.trainSparseRows.clear();
	di.testSparseRows.clear();
	di.featureCountCached = fit.totalInputDims;
	di.expectedCountCached = fit.totalOutputDims;
	di.trainingOHEMaps = fit.oheByCol;
	di.testingOHEMaps = fit.oheByCol;
	di.trainingFeatureIsCategorical = fit.isCategorical;
	di.testingFeatureIsCategorical = fit.isCategorical;
	di.loaded = true;

	// Train/test each model clone on this fold.
	for (size_t i = 0; i < templates.size(); ++i)
	{
		const glades::NNetwork* tmpl = templates[i];
		if (!tmpl || !tmpl->getNNInfo())
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "crossValidate: modelTemplate is NULL or has NULL NNInfo");

		// Validate output dimensionality matches the model's output layer.
		if (tmpl->getNNInfo()->getOutputLayerSize() != fit.totalOutputDims)
			return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "crossValidate: table output dims do not match model output layer size");

		glades::NNetwork net(tmpl->getNNInfo(), tmpl->getNetType());
		// Copy training knobs.
		net.getTrainingConfigMutable() = tmpl->getTrainingConfig();
		net.getTerminatorMutable() = tmpl->getTerminator();
		net.setSeed(tmpl->getSeed());
		// Ensure each fold is independent in RNG usage but deterministic given base seed.
		// Using a fold-derived seed avoids identical dropout/weight-init across folds.
		if (cfg.seed != 0u)
			net.setSeed(tmpl->getSeed() ^ (cfg.seed + static_cast<uint64_t>(i)));

		{
			const glades::NNetworkStatus st = net.train(&di);
			if (!st.ok())
				return st;
		}
		{
			const glades::NNetworkStatus st = net.test(&di);
			if (!st.ok())
				return st;
		}

		const float acc = net.getAccuracy();
		if (!std::isfinite(acc))
			return glades::NNetworkStatus(glades::NNetworkStatus::INTERNAL_ERROR, "crossValidate: non-finite accuracy");
		outFoldAcc[i] = acc;
	}

	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

} // namespace

glades::CrossValidationConfig::CrossValidationConfig()
    : kFolds(5u),
      shuffle(true),
      seed(1u),
      timeSeries(false),
      stratify(true),
      standardizeFlag(glades::GMath::ZSCORE)
{
}

glades::NNetworkStatus glades::crossValidateTableCSV(const shmea::GTable& input,
                                                     const std::vector<glades::NNetwork*>& modelTemplates,
                                                     const glades::CrossValidationConfig& cfg,
                                                     glades::CrossValidationResults* outResults)
{
	if (outResults)
		*outResults = glades::CrossValidationResults();

	if (modelTemplates.empty())
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "crossValidate: no model templates provided");

	const unsigned int N = input.numberOfRows();
	if (N == 0u)
		return glades::NNetworkStatus(glades::NNetworkStatus::EMPTY_DATA, "crossValidate: input table is empty");

	if (cfg.kFolds < 2u)
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "crossValidate: kFolds must be >= 2");

	std::vector<FoldSplit> splits;
	if (cfg.timeSeries)
		build_walk_forward_splits(N, cfg.kFolds, splits);
	else
		build_kfold_splits(input, cfg.kFolds, cfg.shuffle, cfg.seed, cfg.stratify, splits);

	// Filter out folds that have empty train or empty test.
	std::vector<FoldSplit> usable;
	for (size_t i = 0; i < splits.size(); ++i)
	{
		if (!splits[i].trainIdx.empty() && !splits[i].testIdx.empty())
			usable.push_back(splits[i]);
	}
	if (usable.size() < 1u)
		return glades::NNetworkStatus(glades::NNetworkStatus::INVALID_ARGUMENT, "crossValidate: no usable folds (empty train/test)");

	std::vector<float> sumAcc(modelTemplates.size(), 0.0f);
	std::vector< std::vector<float> > foldAcc;
	foldAcc.assign(modelTemplates.size(), std::vector<float>());
	for (size_t m = 0; m < modelTemplates.size(); ++m)
		foldAcc[m].reserve(usable.size());

	for (size_t fi = 0; fi < usable.size(); ++fi)
	{
		const shmea::GTable trainTbl = table_from_indices(input, usable[fi].trainIdx);
		const shmea::GTable testTbl = table_from_indices(input, usable[fi].testIdx);

		// IMPORTANT: fit output label space on the TRAIN fold only to avoid leaking
		// test-fold labels into preprocessing.
		//
		// If a fold's training split is missing a class, that class will be treated
		// as "unknown" at encode time (all-zeros in the one-hot vector), and the
		// fold may be invalid if the model template's output dimensionality assumes
		// a larger fixed label space.
		std::vector< shmea::GPointer<glades::OHE> > foldOutputOHEByCol;
		fit_output_label_ohe_on_train_only(trainTbl, foldOutputOHEByCol);

		std::vector<float> fold;
		glades::CrossValidationConfig cfgFold = cfg;
		// Derive a per-fold seed for deterministic but distinct fold RNG behavior.
		cfgFold.seed = cfg.seed ^ (0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(fi));

		const glades::NNetworkStatus st = run_one_fold(trainTbl, testTbl, modelTemplates, cfgFold, foldOutputOHEByCol, fold);
		if (!st.ok())
			return st;

		for (size_t m = 0; m < fold.size(); ++m)
		{
			sumAcc[m] += fold[m];
			foldAcc[m].push_back(fold[m]);
		}
	}

	if (outResults)
	{
		outResults->totalRows = N;
		outResults->foldsUsed = static_cast<unsigned int>(usable.size());
		outResults->foldTestAccuracy = foldAcc;
		outResults->meanTestAccuracy.assign(modelTemplates.size(), 0.0f);
		for (size_t m = 0; m < modelTemplates.size(); ++m)
			outResults->meanTestAccuracy[m] = static_cast<float>(sumAcc[m] / static_cast<double>(usable.size()));
	}

	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

glades::NNetworkStatus glades::trainTestTableCSV(const shmea::GTable& trainTbl,
                                                 const shmea::GTable& testTbl,
                                                 const std::vector<glades::NNetwork*>& modelTemplates,
                                                 const glades::CrossValidationConfig& cfg,
                                                 std::vector<float>* outTestAccuracies)
{
	// IMPORTANT: fit output label space on TRAIN only (no test leakage).
	std::vector< shmea::GPointer<glades::OHE> > outputOHEByCol;
	fit_output_label_ohe_on_train_only(trainTbl, outputOHEByCol);

	std::vector<float> fold;
	const glades::NNetworkStatus st = run_one_fold(trainTbl, testTbl, modelTemplates, cfg, outputOHEByCol, fold);
	if (!st.ok())
		return st;
	if (outTestAccuracies)
		*outTestAccuracies = fold;
	return glades::NNetworkStatus(glades::NNetworkStatus::OK, std::string());
}

