// Centralized tabular preprocessing implementation.
#include "tabular_preprocessing.h"

#include "../GMath/gmath.h"
#include "Backend/Database/GType.h"

#include <cmath>
#include <limits>
#include <stdio.h> // sprintf

using namespace glades;

namespace {

static inline bool is_finite_float(float x)
{
	return std::isfinite(static_cast<double>(x)) != 0;
}

static bool is_string_cell(const shmea::GType& cell)
{
	return cell.getType() == shmea::GType::STRING_TYPE;
}

static bool infer_is_categorical(const shmea::GTable& t, unsigned int col, const glades::TabularFitOptions& opt)
{
	if (t.numberOfRows() == 0u || col >= t.numberOfCols())
		return false;

	if (opt.inferMode == glades::TabularFitOptions::SCAN_ROWS_STRING)
	{
		unsigned int maxR = opt.scanRows;
		if (maxR == 0u)
			maxR = 1u;
		if (maxR > t.numberOfRows())
			maxR = t.numberOfRows();
		for (unsigned int r = 0; r < maxR; ++r)
		{
			if (is_string_cell(t.getCell(r, col)))
				return true;
		}
		return false;
	}

	// Default: match historical behavior (first row decides).
	return is_string_cell(t.getCell(0u, col));
}

static void welford_fit_numeric(const shmea::GTable& t,
                               unsigned int col,
                               glades::TabularFit::NumericStats& outStats,
                               float& outMin,
                               float& outMax)
{
	const unsigned int R = t.numberOfRows();
	outMin = 0.0f;
	outMax = 0.0f;
	outStats = glades::TabularFit::NumericStats();
	if (R == 0u)
		return;

	// NOTE: We treat non-finite values (NaN/Inf) as missing and skip them when fitting
	// statistics. This avoids corrupting scalers in hostile/dirty real-world data.
	float mn = 0.0f;
	float mx = 0.0f;
	bool haveMinMax = false;
	double mean = 0.0;
	double m2 = 0.0;
	unsigned long long count = 0ULL; // finite count
	unsigned long long missing = 0ULL;
	for (unsigned int r = 0; r < R; ++r)
	{
		const float x = t.getCell(r, col).getFloat();
		if (!is_finite_float(x))
		{
			++missing;
			continue;
		}
		if (!haveMinMax)
		{
			mn = x;
			mx = x;
			haveMinMax = true;
		}
		else
		{
			if (x < mn) mn = x;
			if (x > mx) mx = x;
		}
		++count;
		const double xd = static_cast<double>(x);
		const double delta = xd - mean;
		mean += delta / static_cast<double>(count);
		const double delta2 = xd - mean;
		m2 += delta * delta2;
	}

	float stdev = 0.0f;
	if (count > 1ULL)
	{
		const double var = m2 / static_cast<double>(count - 1ULL);
		if (var > 0.0)
			stdev = static_cast<float>(sqrt(var));
	}

	outStats.finiteCount = count;
	outStats.missingCount = missing;

	if (!haveMinMax || count == 0ULL)
	{
		// All values missing / non-finite. Keep a safe, neutral scaler.
		outStats.minv = 0.0f;
		outStats.maxv = 0.0f;
		outStats.mean = 0.0;
		outStats.stdev = 0.0f;
		outMin = 0.0f;
		outMax = 0.0f;
		return;
	}

	outStats.minv = mn;
	outStats.maxv = mx;
	outStats.mean = mean;
	outStats.stdev = stdev;
	outMin = mn;
	outMax = mx;
}

static float maybe_scale_numeric(float x,
                                const glades::TabularFit::NumericStats& s,
                                int standardizeFlag,
                                bool changeValues)
{
	if (!changeValues)
		return x;

	// Treat NaN/Inf as missing: impute to mean (or 0 if mean is not finite).
	if (!is_finite_float(x))
	{
		const double m = s.mean;
		x = (std::isfinite(m) ? static_cast<float>(m) : 0.0f);
	}

	if (standardizeFlag == glades::GMath::MINMAX)
	{
		const float range = s.maxv - s.minv;
		if (range != 0.0f)
			return (x - s.minv) / range;
		// Constant column: emit a neutral in-range value.
		return 0.0f;
	}
	else if (standardizeFlag == glades::GMath::ZSCORE)
	{
		if (s.stdev != 0.0f)
			return static_cast<float>((static_cast<double>(x) - s.mean) / static_cast<double>(s.stdev));
		// Constant column: emit mean-centered value.
		return 0.0f;
	}

	return x;
}

static void set_err(std::string* errMsg, const char* msg)
{
	if (errMsg)
		*errMsg = (msg ? std::string(msg) : std::string());
}

} // namespace

bool glades::tabularFitOnTrain(const shmea::GTable& train,
                              glades::TabularFit& outFit,
                              const glades::TabularFitOptions& opt,
                              std::string* errMsg)
{
	outFit = glades::TabularFit();

	const unsigned int R = train.numberOfRows();
	const unsigned int C = train.numberOfCols();
	if (R == 0u || C == 0u)
	{
		set_err(errMsg, "tabularFitOnTrain: empty table");
		return false;
	}

	outFit.cols = C;
	outFit.isOutput.assign(C, false);
	outFit.isCategorical.assign(C, false);
	outFit.oheByCol.assign(C, shmea::GPointer<glades::OHE>(new glades::OHE()));
	outFit.numeric.assign(C, glades::TabularFit::NumericStats());
	outFit.colDim.assign(C, 1u);
	outFit.inputOffset.assign(C, 0u);
	outFit.outputOffset.assign(C, 0u);
	outFit.totalInputDims = 0u;
	outFit.totalOutputDims = 0u;
	outFit.globalMin = std::numeric_limits<float>::max();
	outFit.globalMax = -std::numeric_limits<float>::max();
	outFit.sawNumeric = false;

	for (unsigned int c = 0; c < C; ++c)
	{
		const bool isOut = train.isOutput(c);
		outFit.isOutput[c] = isOut;

		const bool isCat = infer_is_categorical(train, c, opt);
		outFit.isCategorical[c] = isCat;

		if (isCat)
		{
			// Output categorical columns may use a globally-fitted label space (e.g., CV).
			if (isOut && opt.globalOutputOHEByCol && (c < opt.globalOutputOHEByCol->size()) && (*opt.globalOutputOHEByCol)[c])
			{
				outFit.oheByCol[c] = (*opt.globalOutputOHEByCol)[c];
			}
			else
			{
				outFit.oheByCol[c]->mapFeatureSpace(train, static_cast<int>(c));
			}

			const unsigned int dim = static_cast<unsigned int>(outFit.oheByCol[c]->size());
			outFit.colDim[c] = dim;
		}
		else
		{
			float mn = 0.0f, mx = 0.0f;
			glades::TabularFit::NumericStats ns;
			welford_fit_numeric(train, c, ns, mn, mx);
			outFit.numeric[c] = ns;
			outFit.colDim[c] = 1u;
			outFit.sawNumeric = true;
			if (mn < outFit.globalMin) outFit.globalMin = mn;
			if (mx > outFit.globalMax) outFit.globalMax = mx;
		}

		// Offset accounting.
		const unsigned int dim = outFit.colDim[c];
		if (isOut)
		{
			outFit.outputOffset[c] = outFit.totalOutputDims;
			outFit.totalOutputDims += dim;
		}
		else
		{
			outFit.inputOffset[c] = outFit.totalInputDims;
			outFit.totalInputDims += dim;
		}
	}

	if (!outFit.sawNumeric)
	{
		outFit.globalMin = 0.0f;
		outFit.globalMax = 0.0f;
	}

	if (outFit.totalInputDims == 0u)
	{
		set_err(errMsg, "tabularFitOnTrain: inferred 0 input dimensions");
		return false;
	}
	if (outFit.totalOutputDims == 0u)
	{
		set_err(errMsg, "tabularFitOnTrain: inferred 0 output dimensions");
		return false;
	}

	return true;
}

bool glades::tabularTransformTrainTest(const shmea::GTable& train,
                                      const shmea::GTable* test,
                                      const glades::TabularFit& fit,
                                      const glades::TabularEncodeOptions& encOpt,
                                      glades::TabularEncoded& out,
                                      std::string* errMsg)
{
	out = glades::TabularEncoded();

	const unsigned int Rtr = train.numberOfRows();
	const unsigned int C = train.numberOfCols();
	const unsigned int Rte = (test ? test->numberOfRows() : 0u);

	if (Rtr == 0u || C == 0u)
	{
		set_err(errMsg, "tabularTransformTrainTest: empty train table");
		return false;
	}
	if (fit.cols != C)
	{
		set_err(errMsg, "tabularTransformTrainTest: fit column count mismatch");
		return false;
	}
	if (test && test->numberOfCols() != C)
	{
		set_err(errMsg, "tabularTransformTrainTest: test column count mismatch vs train");
		return false;
	}

	const unsigned int inDims = fit.totalInputDims;
	const unsigned int outDims = fit.totalOutputDims;
	if (inDims == 0u || outDims == 0u)
	{
		set_err(errMsg, "tabularTransformTrainTest: invalid encoded dimensions");
		return false;
	}

	// Dense categorical encoding semantics are strict one-hot:
	// - baseline is 0.0
	// - hot value is 1.0
	shmea::GVector<float> baseX(inDims, 0.0f);
	shmea::GVector<float> baseY(outDims, 0.0f);

	// Allocate outputs.
	if (encOpt.emitDenseInputs)
		out.trainX = shmea::GMatrix(Rtr, baseX);
	else
		out.trainX.clear();
	if (encOpt.emitDenseOutputs)
		out.trainY = shmea::GMatrix(Rtr, baseY);
	else
		out.trainY.clear();

	if (test)
	{
		if (encOpt.emitDenseInputs)
			out.testX = shmea::GMatrix(Rte, baseX);
		else
			out.testX.clear();
		if (encOpt.emitDenseOutputs)
			out.testY = shmea::GMatrix(Rte, baseY);
		else
			out.testY.clear();
	}

	// Allocate sparse rows if requested (inputs only).
	if (encOpt.emitSparseInputs)
	{
		out.trainSparseX.assign(Rtr, glades::TabularSparseRow());
		if (test)
			out.testSparseX.assign(Rte, glades::TabularSparseRow());

		// Best-effort reserve: (#numeric input cols) + (#categorical input cols).
		unsigned int numericInputs = 0u;
		unsigned int categoricalInputs = 0u;
		for (unsigned int c = 0; c < C; ++c)
		{
			if (fit.isOutput[c])
				continue;
			if (fit.isCategorical[c])
				++categoricalInputs;
			else
				++numericInputs;
		}
		const unsigned int reserveNNZ = numericInputs + categoricalInputs;
		for (unsigned int r = 0; r < Rtr; ++r)
		{
			out.trainSparseX[r].idx.reserve(reserveNNZ);
			out.trainSparseX[r].val.reserve(reserveNNZ);
		}
		if (test)
		{
			for (unsigned int r = 0; r < Rte; ++r)
			{
				out.testSparseX[r].idx.reserve(reserveNNZ);
				out.testSparseX[r].val.reserve(reserveNNZ);
			}
		}
	}

	// Fill train and test.
	for (unsigned int c = 0; c < C; ++c)
	{
		const bool isOut = fit.isOutput[c];
		const bool isCat = fit.isCategorical[c];

		if (isCat)
		{
			const shmea::GPointer<glades::OHE>& ohe = fit.oheByCol[c];
			const unsigned int dim = fit.colDim[c];
			if (!ohe || dim == 0u)
				continue;

			const unsigned int off = isOut ? fit.outputOffset[c] : fit.inputOffset[c];

			// Train
			for (unsigned int r = 0; r < Rtr; ++r)
			{
				const shmea::GType cell = train.getCell(r, c);
				const shmea::GString s = cell.c_str();
				const int idx = ohe->indexAt(s);
				if (idx >= 0 && static_cast<unsigned int>(idx) < dim)
				{
					const unsigned int j = static_cast<unsigned int>(idx);
					if (isOut)
					{
						if (encOpt.emitDenseOutputs)
							out.trainY[r][off + j] = 1.0f;
					}
					else
					{
						if (encOpt.emitDenseInputs)
							out.trainX[r][off + j] = 1.0f;
						if (encOpt.emitSparseInputs)
						{
							out.trainSparseX[r].idx.push_back(off + j);
							out.trainSparseX[r].val.push_back(1.0f);
						}
					}
				}
			}

			// Test (unknown => keep baseline/zeros)
			if (test)
			{
				for (unsigned int r = 0; r < Rte; ++r)
				{
					const shmea::GType cell = test->getCell(r, c);
					const shmea::GString s = cell.c_str();
					const int idx = ohe->indexAt(s);
					if (idx >= 0 && static_cast<unsigned int>(idx) < dim)
					{
						const unsigned int j = static_cast<unsigned int>(idx);
						if (isOut)
						{
							if (encOpt.emitDenseOutputs)
								out.testY[r][off + j] = 1.0f;
						}
						else
						{
							if (encOpt.emitDenseInputs)
								out.testX[r][off + j] = 1.0f;
							if (encOpt.emitSparseInputs)
							{
								out.testSparseX[r].idx.push_back(off + j);
								out.testSparseX[r].val.push_back(1.0f);
							}
						}
					}
				}
			}
		}
		else
		{
			// Numeric.
			const glades::TabularFit::NumericStats& s = fit.numeric[c];
			const unsigned int off = isOut ? fit.outputOffset[c] : fit.inputOffset[c];

			// Train
			for (unsigned int r = 0; r < Rtr; ++r)
			{
				float x = train.getCell(r, c).getFloat();
				x = maybe_scale_numeric(x, s, encOpt.standardizeFlag, encOpt.changeValues);
				if (isOut)
				{
					if (encOpt.emitDenseOutputs)
						out.trainY[r][off] = x;
				}
				else
				{
					if (encOpt.emitDenseInputs)
						out.trainX[r][off] = x;
					if (encOpt.emitSparseInputs && x != 0.0f)
					{
						out.trainSparseX[r].idx.push_back(off);
						out.trainSparseX[r].val.push_back(x);
					}
				}
			}

			// Test
			if (test)
			{
				for (unsigned int r = 0; r < Rte; ++r)
				{
					float x = test->getCell(r, c).getFloat();
					x = maybe_scale_numeric(x, s, encOpt.standardizeFlag, encOpt.changeValues);
					if (isOut)
					{
						if (encOpt.emitDenseOutputs)
							out.testY[r][off] = x;
					}
					else
					{
						if (encOpt.emitDenseInputs)
							out.testX[r][off] = x;
						if (encOpt.emitSparseInputs && x != 0.0f)
						{
							out.testSparseX[r].idx.push_back(off);
							out.testSparseX[r].val.push_back(x);
						}
					}
				}
			}
		}
	}

	return true;
}

