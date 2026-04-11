// Net-type-specific SGD: DFF tensor path (split out of network.cpp)
#include "network.h"
#include "param_layout.h"
#include "sgd_utils.h"

#include "Backend/Database/GLogger.h"

#include "../DataObjects/DataInput.h"
#include "../GMath/gmath.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <time.h>
#include <vector>

#include "logfmt_utils.h"

using namespace glades;
using namespace glades::logfmt;

namespace {
static timespec monotonic_now()
{
	timespec ts;
	ts.tv_sec = 0;
	ts.tv_nsec = 0;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts;
}

static double monotonic_elapsed_ns(const timespec& start, const timespec& end)
{
	const double sec = static_cast<double>(end.tv_sec - start.tv_sec) * 1.0e9;
	const double nsec = static_cast<double>(end.tv_nsec - start.tv_nsec);
	return sec + nsec;
}

static bool invert_small_dense_row_major(const std::vector<float>& matrix,
                                         unsigned int dim,
                                         float ridge,
                                         std::vector<float>& inverseOut)
{
	if (dim == 0u || matrix.size() < static_cast<size_t>(dim) * static_cast<size_t>(dim))
		return false;

	const unsigned int augCols = dim * 2u;
	std::vector<double> aug(static_cast<size_t>(dim) * static_cast<size_t>(augCols), 0.0);
	for (unsigned int r = 0; r < dim; ++r)
	{
		for (unsigned int c = 0; c < dim; ++c)
		{
			double v = static_cast<double>(matrix[static_cast<size_t>(r) * static_cast<size_t>(dim) + c]);
			if (r == c)
				v += static_cast<double>(ridge);
			aug[static_cast<size_t>(r) * static_cast<size_t>(augCols) + c] = v;
		}
		aug[static_cast<size_t>(r) * static_cast<size_t>(augCols) + (dim + r)] = 1.0;
	}

	for (unsigned int col = 0; col < dim; ++col)
	{
		unsigned int pivot = col;
		double pivotAbs =
		    std::fabs(aug[static_cast<size_t>(pivot) * static_cast<size_t>(augCols) + col]);
		for (unsigned int row = col + 1u; row < dim; ++row)
		{
			const double candAbs =
			    std::fabs(aug[static_cast<size_t>(row) * static_cast<size_t>(augCols) + col]);
			if (candAbs > pivotAbs)
			{
				pivot = row;
				pivotAbs = candAbs;
			}
		}
		if (!(pivotAbs > 1e-12))
			return false;
		if (pivot != col)
		{
			for (unsigned int c = 0; c < augCols; ++c)
				std::swap(aug[static_cast<size_t>(pivot) * static_cast<size_t>(augCols) + c],
				          aug[static_cast<size_t>(col) * static_cast<size_t>(augCols) + c]);
		}
		const double invPivot =
		    1.0 / aug[static_cast<size_t>(col) * static_cast<size_t>(augCols) + col];
		for (unsigned int c = 0; c < augCols; ++c)
			aug[static_cast<size_t>(col) * static_cast<size_t>(augCols) + c] *= invPivot;
		for (unsigned int row = 0; row < dim; ++row)
		{
			if (row == col)
				continue;
			const double scale = aug[static_cast<size_t>(row) * static_cast<size_t>(augCols) + col];
			if (std::fabs(scale) <= 1e-18)
				continue;
			for (unsigned int c = 0; c < augCols; ++c)
			{
				aug[static_cast<size_t>(row) * static_cast<size_t>(augCols) + c] -=
				    scale * aug[static_cast<size_t>(col) * static_cast<size_t>(augCols) + c];
			}
		}
	}

	inverseOut.assign(static_cast<size_t>(dim) * static_cast<size_t>(dim), 0.0f);
	for (unsigned int r = 0; r < dim; ++r)
	{
		for (unsigned int c = 0; c < dim; ++c)
		{
			inverseOut[static_cast<size_t>(r) * static_cast<size_t>(dim) + c] =
			    static_cast<float>(aug[static_cast<size_t>(r) * static_cast<size_t>(augCols) + (dim + c)]);
		}
	}
	return true;
}

static void extract_top_singular_modes_row_major(const std::vector<float>& matrix,
                                                 unsigned int rows,
                                                 unsigned int cols,
                                                 unsigned int rank,
                                                 std::vector<float>& sigmaOut,
                                                 std::vector<float>& leftOut,
                                                 std::vector<float>& rightOut)
{
	sigmaOut.assign(rank, 0.0f);
	leftOut.assign(static_cast<size_t>(rank) * static_cast<size_t>(rows), 0.0f);
	rightOut.assign(static_cast<size_t>(rank) * static_cast<size_t>(cols), 0.0f);
	if (rows == 0u || cols == 0u || rank == 0u
	    || matrix.size() < static_cast<size_t>(rows) * static_cast<size_t>(cols))
		return;

	std::vector<float> gram(static_cast<size_t>(rows) * static_cast<size_t>(rows), 0.0f);
	for (unsigned int r1 = 0; r1 < rows; ++r1)
	{
		for (unsigned int r2 = 0; r2 < rows; ++r2)
		{
			double accum = 0.0;
			const size_t off1 = static_cast<size_t>(r1) * static_cast<size_t>(cols);
			const size_t off2 = static_cast<size_t>(r2) * static_cast<size_t>(cols);
			for (unsigned int c = 0; c < cols; ++c)
				accum += static_cast<double>(matrix[off1 + c]) * static_cast<double>(matrix[off2 + c]);
			gram[static_cast<size_t>(r1) * static_cast<size_t>(rows) + r2] = static_cast<float>(accum);
		}
	}

	for (unsigned int m = 0; m < rank; ++m)
	{
		std::vector<float> leftVec(rows, 0.0f);
		leftVec[(m < rows) ? m : 0u] = 1.0f;
		bool valid = true;
		for (unsigned int iter = 0; iter < 6u && valid; ++iter)
		{
			std::vector<float> nextVec(rows, 0.0f);
			for (unsigned int r = 0; r < rows; ++r)
			{
				double accum = 0.0;
				for (unsigned int c = 0; c < rows; ++c)
					accum += static_cast<double>(gram[static_cast<size_t>(r) * static_cast<size_t>(rows) + c])
					      * static_cast<double>(leftVec[c]);
				nextVec[r] = static_cast<float>(accum);
			}
			for (unsigned int pm = 0; pm < m; ++pm)
			{
				const size_t prevOff = static_cast<size_t>(pm) * static_cast<size_t>(rows);
				double proj = 0.0;
				for (unsigned int r = 0; r < rows; ++r)
					proj += static_cast<double>(nextVec[r]) * static_cast<double>(leftOut[prevOff + r]);
				for (unsigned int r = 0; r < rows; ++r)
					nextVec[r] -= static_cast<float>(proj * static_cast<double>(leftOut[prevOff + r]));
			}
			double normSq = 0.0;
			for (unsigned int r = 0; r < rows; ++r)
				normSq += static_cast<double>(nextVec[r]) * static_cast<double>(nextVec[r]);
			if (!(normSq > 1e-18))
			{
				valid = false;
				break;
			}
			const float invNorm = 1.0f / static_cast<float>(sqrt(normSq));
			for (unsigned int r = 0; r < rows; ++r)
				leftVec[r] = nextVec[r] * invNorm;
		}
		if (!valid)
			continue;

		std::vector<float> gramLeft(rows, 0.0f);
		for (unsigned int r = 0; r < rows; ++r)
		{
			double accum = 0.0;
			for (unsigned int c = 0; c < rows; ++c)
				accum += static_cast<double>(gram[static_cast<size_t>(r) * static_cast<size_t>(rows) + c])
				      * static_cast<double>(leftVec[c]);
			gramLeft[r] = static_cast<float>(accum);
		}
		double lambda = 0.0;
		for (unsigned int r = 0; r < rows; ++r)
			lambda += static_cast<double>(leftVec[r]) * static_cast<double>(gramLeft[r]);
		const float sigma = static_cast<float>(sqrt(std::max(0.0, lambda)));
		if (!(sigma > 1e-12f))
			continue;

		const size_t leftOff = static_cast<size_t>(m) * static_cast<size_t>(rows);
		for (unsigned int r = 0; r < rows; ++r)
			leftOut[leftOff + r] = leftVec[r];
		sigmaOut[m] = sigma;

		std::vector<float> rightVec(cols, 0.0f);
		for (unsigned int c = 0; c < cols; ++c)
		{
			double accum = 0.0;
			for (unsigned int r = 0; r < rows; ++r)
				accum += static_cast<double>(matrix[static_cast<size_t>(r) * static_cast<size_t>(cols) + c])
				      * static_cast<double>(leftVec[r]);
			rightVec[c] = static_cast<float>(accum / sigma);
		}
		for (unsigned int pm = 0; pm < m; ++pm)
		{
			const size_t prevOff = static_cast<size_t>(pm) * static_cast<size_t>(cols);
			double proj = 0.0;
			for (unsigned int c = 0; c < cols; ++c)
				proj += static_cast<double>(rightVec[c]) * static_cast<double>(rightOut[prevOff + c]);
			for (unsigned int c = 0; c < cols; ++c)
				rightVec[c] -= static_cast<float>(proj * static_cast<double>(rightOut[prevOff + c]));
		}
		double rightNormSq = 0.0;
		for (unsigned int c = 0; c < cols; ++c)
			rightNormSq += static_cast<double>(rightVec[c]) * static_cast<double>(rightVec[c]);
		if (!(rightNormSq > 1e-18))
		{
			sigmaOut[m] = 0.0f;
			for (unsigned int r = 0; r < rows; ++r)
				leftOut[leftOff + r] = 0.0f;
			continue;
		}
		const float invNorm = 1.0f / static_cast<float>(sqrt(rightNormSq));
		const size_t rightOff = static_cast<size_t>(m) * static_cast<size_t>(cols);
		for (unsigned int c = 0; c < cols; ++c)
			rightOut[rightOff + c] = rightVec[c] * invNorm;
	}
}
} // namespace

void glades::NNetwork::SGDHelper_DFF(unsigned int inputRowCounter, int runType)
{
	using namespace glades::sgd_detail;
	using namespace glades::param_layout;

	const bool isTrain = (runType == RUN_TRAIN);
	const unsigned int dataSize = isTrain ? (di ? di->getTrainSize() : 0u) : (di ? di->getTestSize() : 0u);
	const float gradClip = trainingConfig.perElementGradClip;

	// Ensure tensors exist (modern/tensor-only build).
	if (!ensureTensorParametersInitialized())
	{
		running = false;
		return;
	}
	if (!tensorDff.initialized)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_DFF: tensor state is not initialized");
		running = false;
		return;
	}

	TensorDFFState::HelmState& helm = tensorDff.helm;
	TensorDFFState::AsterState& aster = tensorDff.aster;
	const bool helmEnabled =
	    isTrain
	    && (trainingConfig.optimizer.type == OptimizerConfig::ATLAS)
	    && trainingConfig.atlas.helmEnabled
	    && helm.initialized
	    && !tensorDff.T.empty();
	const bool asterEnabled =
	    isTrain
	    && (trainingConfig.optimizer.type == OptimizerConfig::ATLAS)
	    && trainingConfig.atlas.asterEnabled
	    && aster.initialized
	    && !tensorDff.T.empty();
	if (helmEnabled && tensorDff.batchCount == 0u)
	{
		std::fill(helm.batchHiddenSum.begin(), helm.batchHiddenSum.end(), 0.0f);
		std::fill(helm.batchHiddenSqSum.begin(), helm.batchHiddenSqSum.end(), 0.0f);
		std::fill(helm.batchResidualSum.begin(), helm.batchResidualSum.end(), 0.0f);
		std::fill(helm.batchResidualSqSum.begin(), helm.batchResidualSqSum.end(), 0.0f);
	}
	if (asterEnabled && tensorDff.batchCount == 0u)
	{
		std::fill(aster.batchHiddenSum.begin(), aster.batchHiddenSum.end(), 0.0f);
		std::fill(aster.batchHiddenSqSum.begin(), aster.batchHiddenSqSum.end(), 0.0f);
		std::fill(aster.batchResidualSum.begin(), aster.batchResidualSum.end(), 0.0f);
		std::fill(aster.batchResidualSqSum.begin(), aster.batchResidualSqSum.end(), 0.0f);
	}

	// Input row: prefer sparse view when available.
	const unsigned int in = tensorDff.sizes.empty() ? 0u : tensorDff.sizes[0];
	const unsigned int* xIdx = NULL;
	const float* xVal = NULL;
	unsigned int xNNZ = 0u;
	unsigned int xFullSize = 0u;
	bool haveSparseX = false;
	if (di)
	{
		if (isTrain)
			haveSparseX = di->getTrainRowSparseView(inputRowCounter, xIdx, xVal, xNNZ, xFullSize);
		else
			haveSparseX = di->getTestRowSparseView(inputRowCounter, xIdx, xVal, xNNZ, xFullSize);
	}
	if (!(haveSparseX && xFullSize == in))
	{
		haveSparseX = false;
		xIdx = NULL;
		xVal = NULL;
		xNNZ = 0u;
		xFullSize = 0u;
	}
	// Filtered sparse features after (optional) input dropout.
	std::vector<unsigned int> xIdxF;
	std::vector<float> xValF;

	// Dropout masks (tensor-only):
	// - Layer 0 uses skeleton->getPInput()
	// - Hidden layers use skeleton->getPDropout(hiddenIdx)
	// Output layer is never dropped.
	//
	// IMPORTANT: This uses *inverted dropout* semantics:
	// - During training, kept activations are scaled by 1/(1-p) so their expectation matches inference.
	// - During evaluation, dropout is disabled (no scaling).
	const int H = skeleton->numHiddenLayers();
	std::vector<std::vector<unsigned char> > keepMasks;
	keepMasks.resize(static_cast<size_t>(H) + 1u);
	std::vector<float> keepScales;
	keepScales.assign(keepMasks.size(), 1.0f);
	{
		const float pIn = skeleton->getPInput();
		// Sparse input: never allocate a full per-feature dropout mask.
		// We will apply dropout only to the active (nnz) indices.
		if (!haveSparseX)
			keepMasks[0].assign(in, 1u);
		else
			keepMasks[0].clear();
		if (isTrain && pIn > 0.0f && !haveSparseX)
		{
			// Inverted dropout: scale kept activations by 1/(1-p).
			// Guard p>=1 to avoid division-by-zero (all units will be dropped anyway).
			if (pIn < 1.0f)
				keepScales[0] = 1.0f / (1.0f - pIn);
			for (unsigned int i = 0; i < in; ++i)
				keepMasks[0][i] = (glades::rng::uniform_double(rngEngine, 0.0, 1.0) >= static_cast<double>(pIn)) ? 1u : 0u;
		}
		// Sparse input dropout is handled after we read xIdx/xVal (see forward pass).
		if (isTrain && pIn > 0.0f && haveSparseX)
		{
			if (pIn < 1.0f)
				keepScales[0] = 1.0f / (1.0f - pIn);
		}
	}
	for (int h = 0; h < H; ++h)
	{
		const unsigned int layerIdx = static_cast<unsigned int>(h) + 1u;
		const unsigned int sz = (layerIdx < tensorDff.sizes.size()) ? tensorDff.sizes[layerIdx] : 0u;
		keepMasks[layerIdx].assign(sz, 1u);
		const float p = skeleton->getPDropout(static_cast<unsigned int>(h));
		if (isTrain && p > 0.0f)
		{
			if (p < 1.0f && layerIdx < keepScales.size())
				keepScales[layerIdx] = 1.0f / (1.0f - p);
			for (unsigned int j = 0; j < sz; ++j)
				keepMasks[layerIdx][j] = (glades::rng::uniform_double(rngEngine, 0.0, 1.0) >= static_cast<double>(p)) ? 1u : 0u;
		}
	}

	// Reset per-sample outputs
	results.clear();
	// Evaluation should not destroy any caller-visible training records.
	if (isTrain)
		nbRecord.clear();

	// Tensor buffers are initialized in ensureTensorParametersInitialized().

	// Forward pass for this sample
	{
		const float* xData = NULL;
		unsigned int xSize = 0u;
		if (!haveSparseX && di)
		{
			if (isTrain)
				di->getTrainRowView(inputRowCounter, xData, xSize);
			else
				di->getTestRowView(inputRowCounter, xData, xSize);
		}
		// Hard safety check: reject non-finite inputs early (sampled for performance on huge rows).
		if (!haveSparseX)
		{
			if (!span_all_finite_bounded(xData, xSize, /*maxChecks*/ 32u))
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_DFF: non-finite input data detected (NaN/Inf)");
				running = false;
				return;
			}
			for (unsigned int i = 0; i < in; ++i)
			{
				const bool keep = (keepMasks[0].empty() ? true : (keepMasks[0][i] != 0u));
				const float x = (xData && (i < xSize)) ? xData[i] : 0.0f;
				tensorDff.a[0][i] = keep ? (x * keepScales[0]) : 0.0f;
			}
		}
		else
		{
			// Sparse: build the post-dropout sparse view for this sample.
			// (Never materialize tensorDff.a[0] as a dense vector.)
			xIdxF.clear();
			xValF.clear();
			xIdxF.reserve(xNNZ);
			xValF.reserve(xNNZ);

			// Validate sparse input values (nnz is expected to be small).
			for (unsigned int k = 0; k < xNNZ; ++k)
			{
				const unsigned int fi = xIdx[k];
				const float xv = xVal[k];
				if (!is_finite(xv))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_DFF: non-finite sparse input value detected (NaN/Inf)");
					running = false;
					return;
				}
				// Enforce bounds defensively.
				if (fi >= in)
					continue;

				// Optional input dropout: apply only to active indices.
				const float pIn = skeleton->getPInput();
				if (isTrain && pIn > 0.0f)
				{
					const bool keep = (glades::rng::uniform_double(rngEngine, 0.0, 1.0) >= static_cast<double>(pIn));
					if (!keep)
						continue;
					xIdxF.push_back(fi);
					xValF.push_back(xv * keepScales[0]);
				}
				else
				{
					xIdxF.push_back(fi);
					xValF.push_back(xv);
				}
			}
		}

		// We record raw pre-activations for GUI visualization on the last sample only.
		const bool recordActivations = (di && (dataSize > 0) && (inputRowCounter == dataSize - 1));

		const int costFx = skeleton->getOutputType();
		const unsigned int outSize = skeleton->getOutputLayerSize();
		const unsigned int lastTransition = (tensorDff.T.size() > 0) ? static_cast<unsigned int>(tensorDff.T.size() - 1) : 0;
		const bool useSoftmax =
		    ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (outSize > 1);
		std::vector<float> outLogits;
		std::vector<float> outProbs;

		for (unsigned int t = 0; t < tensorDff.T.size(); ++t)
		{
			const TensorDFFState::Transition& tr = tensorDff.T[t];

			const int actFx = skeleton->getActivationType(t);
			const float actParam = skeleton->getActivationParam(t);
			const bool softmaxThisLayer = useSoftmax && (t == lastTransition) && (tr.out == outSize);
			if (softmaxThisLayer)
				outLogits.assign(tr.out, 0.0f);

			const bool dropoutThisLayer = isTrain && (t < static_cast<unsigned int>(H));
			const unsigned int outLayerIdx = t + 1u; // layer index in tensorDff.a / keepMasks (hidden only)
			const float dropoutScale =
			    (dropoutThisLayer && outLayerIdx < keepScales.size()) ? keepScales[outLayerIdx] : 1.0f;

			for (unsigned int j = 0; j < tr.out; ++j)
			{
				// Dropout: hidden layers only.
				if (dropoutThisLayer)
				{
					if (outLayerIdx < keepMasks.size() && (j < keepMasks[outLayerIdx].size()) && (keepMasks[outLayerIdx][j] == 0u))
					{
						tensorDff.a[t + 1][j] = 0.0f;
						continue;
					}
				}

				float z = (j < tr.bias.size()) ? tr.bias[j] : 0.0f;
				const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(tr.in);
				if (haveSparseX && t == 0u)
				{
					// Sparse input only applies to the first transition.
					for (unsigned int kk = 0; kk < static_cast<unsigned int>(xIdxF.size()); ++kk)
					{
						const unsigned int fi = xIdxF[kk];
						// fi < tr.in is guaranteed by our filtering above.
						z += tr.W[rowOff + fi] * xValF[kk];
					}
				}
				else
				{
					for (unsigned int i = 0; i < tr.in; ++i)
						z += tr.W[rowOff + i] * tensorDff.a[t][i];
				}
				if (!is_finite(z))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite pre-activation detected (NaN/Inf)");
					running = false;
					return;
				}

				if (recordActivations)
					cNodeActivations.addFloat(z);

				if (softmaxThisLayer)
					outLogits[j] = z;
				else
				{
					const float a = GMath::squash(z, actFx, actParam);
					if (!is_finite(a))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite activation detected (NaN/Inf)");
						running = false;
						return;
					}
					// Inverted dropout scaling on hidden layers during training.
					tensorDff.a[t + 1][j] = dropoutThisLayer ? (a * dropoutScale) : a;
				}
			}

			// Apply softmax as a layer-level activation for multi-class classification/KL.
			if (softmaxThisLayer)
			{
				softmax_stable(outLogits, outProbs);
				for (unsigned int j = 0; j < tr.out; ++j)
				{
					const float p = outProbs[j];
					if (!is_finite(p))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite softmax probability detected (NaN/Inf)");
						running = false;
						return;
					}
					tensorDff.a[t + 1][j] = p;
				}
			}
		}

		if (recordActivations)
			cNodeActivations.addString(",");
	}

	// Output layer bookkeeping (results + loss; accuracy is derived later per output type)
	{
		const unsigned int outSize = skeleton->getOutputLayerSize();
		const unsigned int last = (tensorDff.sizes.size() > 0) ? static_cast<unsigned int>(tensorDff.sizes.size() - 1) : 0;
		const float* yData = NULL;
		unsigned int ySize = 0u;
		if (di)
		{
			if (isTrain)
				di->getTrainExpectedRowView(inputRowCounter, yData, ySize);
			else
				di->getTestExpectedRowView(inputRowCounter, yData, ySize);
		}
		if (!span_all_finite_bounded(yData, ySize, /*maxChecks*/ 32u))
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_DFF: non-finite expected outputs detected (NaN/Inf)");
			running = false;
			return;
		}
		const int costFx = skeleton->getOutputType();
		const bool useSoftmax =
		    ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (outSize > 1);
		const unsigned int N = dataSize;

		double loss = 0.0;
		for (unsigned int k = 0; k < outSize; ++k)
		{
			const float pred = tensorDff.a[last][k];
			const float expv = (yData && (k < ySize)) ? yData[k] : 0.0f;
			if (!is_finite(expv))
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_DFF: non-finite expected value detected (NaN/Inf)");
				running = false;
				return;
			}
			if (!is_finite(pred))
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite prediction detected (NaN/Inf)");
				running = false;
				return;
			}

			results.addFloat(expv);
			results.addFloat(pred);

			if (costFx == GMath::REGRESSION)
			{
				const double diff = static_cast<double>(pred) - static_cast<double>(expv);
				regSSE += diff * diff;
				regSAE += fabs(diff);
				regSumY += static_cast<double>(expv);
				regSumY2 += static_cast<double>(expv) * static_cast<double>(expv);
				++regCount;
			}
			else if (useSoftmax)
			{
				const float p = clamp_prob01(pred);
				if (costFx == GMath::CLASSIFICATION)
					loss += -static_cast<double>(expv) * log(static_cast<double>(p));
				else
					loss += static_cast<double>(GMath::KLDivergence(expv, pred));
			}
			else
			{
				const double denom = (N > 0 && outSize > 0) ? static_cast<double>(N) * static_cast<double>(outSize) : 1.0;
				overallTotalError += static_cast<float>(GMath::outputNodeCost(expv, pred, static_cast<float>(denom), costFx));
			}
		}

		if (useSoftmax && N > 0)
			overallTotalError += static_cast<float>(loss / static_cast<double>(N));
		if (!std::isfinite(overallTotalError))
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite loss aggregate detected (NaN/Inf)");
			running = false;
			return;
		}

		// Add current results to cmatrix for accuracy vars
		if ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL))
		{
			const int expIdx = GMath::argmax(yData, outSize);
			const int predIdx = GMath::argmax(&tensorDff.a[last][0], outSize);
			confusionMatrix.addResultDirect(static_cast<unsigned int>(expIdx), static_cast<unsigned int>(predIdx));
		}
	}

	// Progress logs for long DFF epochs (bounded to ~20 messages per epoch,
	// and at most once per second wall-clock to avoid flooding on fast small datasets).
	{
		shmea::GLogger* logger = getLogger();
		if (logger && dataSize > 0u)
		{
			unsigned int every = 1u;
			if (dataSize > 20u)
				every = dataSize / 20u;
			if (every == 0u)
				every = 1u;
			const unsigned int done = inputRowCounter + 1u;
			if (done == dataSize || (done % every) == 0u)
			{
				const int64_t now = getCurrentTimeMilliseconds();
				if (now - lastStepLogTime >= 1000)
				{
					lastStepLogTime = now;
					std::ostringstream oss;
					oss << "event=nn_step_progress";
					append_logfmt_kv(oss, "net_type", netType);
					append_logfmt_kv(oss, "run_type", std::string(isTrain ? "train" : "eval"));
					append_logfmt_kv(oss, "epoch", epochs);
					append_logfmt_kv(oss, "step", done);
					append_logfmt_kv(oss, "steps_total", dataSize);
					append_logfmt_kv(oss, "loss_so_far", overallTotalError);
					append_logfmt_kv(oss, "lr_mult", lrScheduleMultiplier);
					if (trainingConfig.globalGradClipNorm > 0.0f)
					{
						append_logfmt_kv(oss, "grad_norm", lastGradNorm);
						append_logfmt_kv(oss, "grad_norm_scale", lastGradNormScale);
					}
					else
					{
						append_logfmt_kv(oss, "grad_norm", std::string("na"));
						append_logfmt_kv(oss, "grad_norm_scale", std::string("na"));
					}
					logger->info("NNetwork", shmea::GString(oss.str().c_str()));
				}
			}
		}
	}

	// Backprop + SGD update (minibatched) for train
	if (isTrain)
	{
		const unsigned int numLayers = static_cast<unsigned int>(tensorDff.sizes.size());
		if (numLayers >= 2)
		{
			const unsigned int lastLayer = numLayers - 1;
			const unsigned int lastTransition = static_cast<unsigned int>(tensorDff.T.size() - 1);
			const unsigned int outSize = tensorDff.sizes[lastLayer];
			const float* yData = NULL;
			unsigned int ySize = 0u;
			if (di)
				di->getTrainExpectedRowView(inputRowCounter, yData, ySize);
			const int costFx = skeleton->getOutputType();
			const int outActFx = skeleton->getActivationType(lastTransition);
			const float outActParam = skeleton->getActivationParam(lastTransition);

			// Output deltas
			for (unsigned int k = 0; k < outSize; ++k)
			{
				const float pred = tensorDff.a[lastLayer][k];
					const float expv = (yData && (k < ySize)) ? yData[k] : 0.0f;
				const bool useSoftmax =
				    ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) &&
				    (skeleton->getOutputLayerSize() > 1);
				if (useSoftmax)
				{
					// Softmax + (cross-entropy or KL) => dL/dz = p - y
					tensorDff.delta[lastLayer][k] = clipf_maybe(pred - expv, gradClip);
				}
				else
				{
					// Binary classification: if the output nonlinearity is sigmoid, use the stable
					// combined derivative for BCE-with-sigmoid:
					//   dL/dz = p - y
					// This avoids the numerically-unstable (p-y)/(p(1-p)) * p(1-p) path.
					if ((costFx == GMath::CLASSIFICATION) && (outActFx == GMath::SIGMOID || outActFx == GMath::SIGMOIDP))
					{
						tensorDff.delta[lastLayer][k] = clipf_maybe(pred - expv, gradClip);
					}
					else
					{
						const float dCost_dA = GMath::costErrDer(expv, pred, costFx);
						const float dA_dZ = GMath::activationErrDer(pred, outActFx, outActParam);
						// Basic gradient clipping for stability in extreme cases.
						tensorDff.delta[lastLayer][k] = clipf_maybe(dCost_dA * dA_dZ, gradClip);
					}
				}
				if (!is_finite(tensorDff.delta[lastLayer][k]))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite output delta detected (NaN/Inf)");
					running = false;
					return;
				}
			}

			// Hidden deltas (backwards)
			for (int li = static_cast<int>(lastLayer) - 1; li >= 1; --li)
			{
				const unsigned int l = static_cast<unsigned int>(li);
				const TensorDFFState::Transition& nextTr = tensorDff.T[l]; // maps layer l -> l+1

				const int actFx = skeleton->getActivationType(l - 1);
				const float actParam = skeleton->getActivationParam(l - 1);

				for (unsigned int i = 0; i < tensorDff.sizes[l]; ++i)
				{
					if (l < keepMasks.size() && (i < keepMasks[l].size()) && (keepMasks[l][i] == 0u))
					{
						tensorDff.delta[l][i] = 0.0f;
						continue;
					}

					float sum = 0.0f;
					// sum_j delta_{l+1}[j] * W_next[j,i]
					for (unsigned int j = 0; j < nextTr.out; ++j)
						sum += tensorDff.delta[l + 1][j] * nextTr.W[static_cast<size_t>(j) * static_cast<size_t>(nextTr.in) + i];

					// If dropout was applied to this hidden layer, tensorDff.a[l][i] stores the *scaled*
					// activation. activationErrDer expects the *unscaled* activation output (e.g. tanh(x), sigmoid(x)),
					// so undo the inverted-dropout scale here for correct derivatives.
					float aUnscaled = tensorDff.a[l][i];
					if (isTrain && l < keepScales.size())
					{
						const float sc = keepScales[l];
						if (sc != 1.0f)
						{
							// Keep-mask is non-zero here (dropped units continued above).
							aUnscaled = aUnscaled / sc;
						}
					}
					const float dA_dZ = GMath::activationErrDer(aUnscaled, actFx, actParam);
					// Basic gradient clipping for stability in extreme cases.
					tensorDff.delta[l][i] = clipf_maybe(sum * dA_dZ, gradClip);
					if (!is_finite(tensorDff.delta[l][i]))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite hidden delta detected (NaN/Inf)");
						running = false;
						return;
					}
				}
			}

			// Accumulate minibatch gradients
			for (unsigned int t = 0; t < tensorDff.T.size(); ++t)
			{
				TensorDFFState::Transition& tr = tensorDff.T[t];
				const unsigned int out = tr.out;
				const unsigned int in = tr.in;

				for (unsigned int j = 0; j < out; ++j)
				{
					const float d = tensorDff.delta[t + 1][j];
					if (!is_finite(d))
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite minibatch delta detected (NaN/Inf)");
						running = false;
						return;
					}
					if (j < tr.gBias.size())
						tr.gBias[j] += d;
					const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(in);
					if (haveSparseX && t == 0u)
					{
						// Sparse input: only accumulate gradients for active indices.
						for (unsigned int kk = 0; kk < static_cast<unsigned int>(xIdxF.size()); ++kk)
						{
							const unsigned int fi = xIdxF[kk];
							// fi < in by construction.
							tr.gW[rowOff + fi] += d * xValF[kk];
						}
					}
					else
					{
						for (unsigned int i = 0; i < in; ++i)
							tr.gW[rowOff + i] += d * tensorDff.a[t][i];
					}
				}
			}
			if (helmEnabled)
			{
				const std::vector<float>& outputResidual = tensorDff.delta[lastTransition + 1u];
				const unsigned int outputDim = std::min<unsigned int>(static_cast<unsigned int>(outputResidual.size()), helm.outputDim);
				const unsigned int trackedLayers = std::min<unsigned int>(
				    static_cast<unsigned int>(helm.hiddenLayerActivationIndices.size()),
				    std::min<unsigned int>(static_cast<unsigned int>(helm.hiddenLayerOffsets.size()),
				                           static_cast<unsigned int>(helm.hiddenLayerSizes.size())));
				for (unsigned int l = 0; l < trackedLayers; ++l)
				{
					const unsigned int actIndex = helm.hiddenLayerActivationIndices[l];
					if (actIndex >= tensorDff.a.size())
						continue;
					const unsigned int offset = helm.hiddenLayerOffsets[l];
					const unsigned int hiddenDim = std::min<unsigned int>(
					    helm.hiddenLayerSizes[l],
					    static_cast<unsigned int>(tensorDff.a[actIndex].size()));
					const std::vector<float>& hiddenAct = tensorDff.a[actIndex];
					for (unsigned int i = 0; i < hiddenDim && (offset + i) < helm.batchHiddenSum.size(); ++i)
					{
						const float h = hiddenAct[i];
						helm.batchHiddenSum[offset + i] += h;
						helm.batchHiddenSqSum[offset + i] += h * h;
					}
				}
				for (unsigned int j = 0; j < outputDim; ++j)
				{
					const float e = outputResidual[j];
					helm.batchResidualSum[j] += e;
					helm.batchResidualSqSum[j] += e * e;
				}
			}
			if (asterEnabled)
			{
				const std::vector<float>& outputResidual = tensorDff.delta[lastTransition + 1u];
				const unsigned int outputDim =
				    std::min<unsigned int>(static_cast<unsigned int>(outputResidual.size()), aster.outputDim);
				const unsigned int trackedLayers = std::min<unsigned int>(
				    static_cast<unsigned int>(aster.hiddenLayerActivationIndices.size()),
				    std::min<unsigned int>(static_cast<unsigned int>(aster.hiddenLayerOffsets.size()),
				                           static_cast<unsigned int>(aster.hiddenLayerSizes.size())));
				for (unsigned int l = 0; l < trackedLayers; ++l)
				{
					const unsigned int actIndex = aster.hiddenLayerActivationIndices[l];
					if (actIndex >= tensorDff.a.size())
						continue;
					const unsigned int offset = aster.hiddenLayerOffsets[l];
					const unsigned int hiddenDim = std::min<unsigned int>(
					    aster.hiddenLayerSizes[l],
					    static_cast<unsigned int>(tensorDff.a[actIndex].size()));
					const std::vector<float>& hiddenAct = tensorDff.a[actIndex];
					for (unsigned int i = 0; i < hiddenDim && (offset + i) < aster.batchHiddenSum.size(); ++i)
					{
						const float h = hiddenAct[i];
						aster.batchHiddenSum[offset + i] += h;
						aster.batchHiddenSqSum[offset + i] += h * h;
					}
				}
				for (unsigned int j = 0; j < outputDim; ++j)
				{
					const float e = outputResidual[j];
					aster.batchResidualSum[j] += e;
					aster.batchResidualSqSum[j] += e * e;
				}
			}
			++tensorDff.batchCount;

			// End-of-batch detection
				const unsigned int trainSize = di ? di->getTrainSize() : 0;
				const int effectiveMiniBatchSize = (minibatchSize > 0) ? minibatchSize : static_cast<int>(trainSize);
				const bool isLastSample = (trainSize > 0) && (inputRowCounter + 1 >= trainSize);
			const bool isBatchEnd =
				(effectiveMiniBatchSize <= 1) ||
				(((inputRowCounter + 1) % static_cast<unsigned int>(effectiveMiniBatchSize)) == 0) ||
				isLastSample;

			if (isBatchEnd && tensorDff.batchCount > 0)
			{
				const float invBatch = 1.0f / static_cast<float>(tensorDff.batchCount);

				if (helmEnabled && lastTransition < tensorDff.T.size())
				{
					TensorDFFState::Transition& outTr = tensorDff.T[lastTransition];
					const ATLASConfig& acHelm = trainingConfig.atlas;
					const unsigned int rawHiddenDim = helm.rawHiddenDim;
					const unsigned int hiddenDim = helm.hiddenDim;
					const unsigned int outputDim = std::min<unsigned int>(helm.outputDim, outTr.out);
					const unsigned int pastDim = hiddenDim + outputDim;
					const unsigned int modeRank = std::min<unsigned int>(std::max(1u, helm.modeRank),
					                                                   std::max(1u, outputDim));
					const unsigned int trackedLayers = std::min<unsigned int>(
					    static_cast<unsigned int>(helm.hiddenLayerActivationIndices.size()),
					    std::min<unsigned int>(static_cast<unsigned int>(helm.hiddenLayerOffsets.size()),
					                           static_cast<unsigned int>(helm.hiddenLayerSizes.size())));
					const float varEps = 1e-6f;
					const float sigmaEps = 1e-12f;
					const float betaHelm = std::min<float>(std::max<float>(acHelm.beta, 0.0f), 1.0f);
					std::vector<float> rawHiddenMean(rawHiddenDim, 0.0f);
					std::vector<float> hiddenMean(hiddenDim, 0.0f);
					std::vector<float> residualMean(outputDim, 0.0f);
					std::vector<float> hiddenStd(hiddenDim, 1.0f);
					std::vector<float> residualStd(outputDim, 1.0f);
					std::vector<float> pastSignal(pastDim, 0.0f);
					std::vector<float> currentResidualW(outputDim, 0.0f);
					std::vector<float> predictedResidualW(outputDim, 0.0f);
					std::vector<float> predictedLatent(modeRank, 0.0f);
					std::vector<float> nextSigma(modeRank, 0.0f);
					std::vector<float> nextLeft(static_cast<size_t>(modeRank) * static_cast<size_t>(outputDim), 0.0f);
					std::vector<float> nextRight(static_cast<size_t>(modeRank) * static_cast<size_t>(pastDim), 0.0f);
					std::vector<float> gram(static_cast<size_t>(outputDim) * static_cast<size_t>(outputDim), 0.0f);
					const unsigned int outputHiddenOffset =
					    helm.hiddenLayerOffsets.empty() ? 0u : helm.hiddenLayerOffsets.back();
					const unsigned int outputHiddenDim =
					    (helm.hiddenLayerSizes.empty() || outputHiddenOffset >= rawHiddenDim)
					        ? 0u
					        : std::min<unsigned int>(
					              std::min<unsigned int>(helm.hiddenLayerSizes.back(), outTr.in),
					              rawHiddenDim - outputHiddenOffset);

					for (unsigned int i = 0; i < rawHiddenDim; ++i)
						rawHiddenMean[i] = helm.batchHiddenSum[i] * invBatch;

					for (unsigned int l = 0; l < trackedLayers && ((l + 1u) * outputDim) <= hiddenDim; ++l)
					{
						const unsigned int layerOffset = helm.hiddenLayerOffsets[l];
						const unsigned int layerSize = (layerOffset < rawHiddenDim)
						                             ? std::min<unsigned int>(helm.hiddenLayerSizes[l], rawHiddenDim - layerOffset)
						                             : 0u;
						if (layerSize == 0u)
							continue;

						std::vector<float> transported(layerSize, 0.0f);
						for (unsigned int i = 0; i < layerSize; ++i)
							transported[i] = rawHiddenMean[layerOffset + i];
						unsigned int currentActIndex = helm.hiddenLayerActivationIndices[l];

						for (unsigned int dl = l + 1u; dl < trackedLayers; ++dl)
						{
							if (currentActIndex >= tensorDff.T.size())
							{
								transported.clear();
								break;
							}
							const TensorDFFState::Transition& downTr = tensorDff.T[currentActIndex];
							const unsigned int nextOffset = helm.hiddenLayerOffsets[dl];
							const unsigned int nextSize = (nextOffset < rawHiddenDim)
							                            ? std::min<unsigned int>(helm.hiddenLayerSizes[dl], rawHiddenDim - nextOffset)
							                            : 0u;
							if (nextSize == 0u || transported.empty())
							{
								transported.clear();
								break;
							}

							std::vector<float> nextTransport(nextSize, 0.0f);
							const unsigned int dotIn = std::min<unsigned int>(static_cast<unsigned int>(transported.size()), downTr.in);
							const unsigned int dotOut = std::min<unsigned int>(nextSize, downTr.out);
							const int nextActFx = skeleton->getActivationType(static_cast<int>(currentActIndex));
							const float nextActParam = skeleton->getActivationParam(static_cast<int>(currentActIndex));
							for (unsigned int j = 0; j < dotOut; ++j)
							{
								double accum = 0.0;
								const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(downTr.in);
								for (unsigned int i = 0; i < dotIn; ++i)
									accum += static_cast<double>(downTr.W[rowOff + i]) * static_cast<double>(transported[i]);
								const float nextActivation = rawHiddenMean[nextOffset + j];
								const float gate = GMath::activationErrDer(nextActivation, nextActFx, nextActParam);
								nextTransport[j] = static_cast<float>(accum) * gate;
							}
							transported.swap(nextTransport);
							currentActIndex = helm.hiddenLayerActivationIndices[dl];
						}

						if (transported.empty())
							continue;

						const unsigned int projectedOff = l * outputDim;
						const unsigned int dotIn = std::min<unsigned int>(static_cast<unsigned int>(transported.size()), outTr.in);
						for (unsigned int j = 0; j < outputDim; ++j)
						{
							double accum = 0.0;
							const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(outTr.in);
							for (unsigned int i = 0; i < dotIn; ++i)
								accum += static_cast<double>(outTr.W[rowOff + i]) * static_cast<double>(transported[i]);
							hiddenMean[projectedOff + j] = static_cast<float>(accum);
						}
					}

					for (unsigned int i = 0; i < hiddenDim; ++i)
					{
						const float secondMoment = hiddenMean[i] * hiddenMean[i];
						const float updatedVar = (betaHelm * helm.hiddenVar[i]) + ((1.0f - betaHelm) * std::max(secondMoment, varEps));
						helm.hiddenVar[i] = std::max(updatedVar, varEps);
						hiddenStd[i] = sqrtf(helm.hiddenVar[i]);
						pastSignal[i] = (hiddenStd[i] > sigmaEps) ? (helm.prevHiddenMean[i] / hiddenStd[i]) : 0.0f;
					}
					for (unsigned int j = 0; j < outputDim; ++j)
					{
						residualMean[j] = helm.batchResidualSum[j] * invBatch;
						const float secondMoment = helm.batchResidualSqSum[j] * invBatch;
						const float updatedVar = (betaHelm * helm.residualVar[j]) + ((1.0f - betaHelm) * std::max(secondMoment, varEps));
						helm.residualVar[j] = std::max(updatedVar, varEps);
						residualStd[j] = sqrtf(helm.residualVar[j]);
						currentResidualW[j] = (residualStd[j] > sigmaEps) ? (residualMean[j] / residualStd[j]) : 0.0f;
						pastSignal[hiddenDim + j] = (residualStd[j] > sigmaEps) ? (helm.prevResidualMean[j] / residualStd[j]) : 0.0f;
					}

					double effectiveSigmaSq = 0.0;
					for (unsigned int m = 0; m < modeRank; ++m)
					{
						const float modeSigma = (m < helm.sigma.size()) ? std::max(0.0f, helm.sigma[m]) : 0.0f;
						const float modePole = (m < helm.pole.size()) ? helm.pole[m] : 0.0f;
						const float modeLatent = (m < helm.latent.size()) ? helm.latent[m] : 0.0f;
						double latentInput = 0.0;
						const size_t rightOff = static_cast<size_t>(m) * static_cast<size_t>(pastDim);
						for (unsigned int p = 0; p < pastDim && (rightOff + p) < helm.rightMode.size(); ++p)
							latentInput += static_cast<double>(helm.rightMode[rightOff + p]) * static_cast<double>(pastSignal[p]);
						predictedLatent[m] = static_cast<float>(
						    static_cast<double>(modePole) * static_cast<double>(modeLatent) + latentInput);
						effectiveSigmaSq += static_cast<double>(modeSigma) * static_cast<double>(modeSigma);
						const size_t leftOff = static_cast<size_t>(m) * static_cast<size_t>(outputDim);
						for (unsigned int j = 0; j < outputDim && (leftOff + j) < helm.leftMode.size(); ++j)
							predictedResidualW[j] += modeSigma * helm.leftMode[leftOff + j] * predictedLatent[m];
					}
					const float effectiveSigma = static_cast<float>(sqrt(effectiveSigmaSq));

					double targetNormSq = 0.0;
					double errNormSq = 0.0;
					for (unsigned int j = 0; j < outputDim; ++j)
					{
						const double target = static_cast<double>(currentResidualW[j]);
						const double err = target - static_cast<double>(predictedResidualW[j]);
						targetNormSq += target * target;
						errNormSq += err * err;
					}
					float helmPredR2 = 0.0f;
					if (targetNormSq > 1e-12)
						helmPredR2 = static_cast<float>(std::max<double>(0.0, 1.0 - (errNormSq / targetNormSq)));
					const float helmEdge = std::max(0.0f, effectiveSigma);
					float helmTransferScale = 0.0f;
					if (helmEdge > acHelm.helmEdgeThreshold)
						helmTransferScale = (helmEdge - acHelm.helmEdgeThreshold) / (helmEdge + 1e-6f);
					float helmMemoryGain = acHelm.helmMemoryScale
					                     * std::max(0.0f, helmTransferScale)
					                     * std::max(0.0f, helmPredR2);
					if (!is_finite(helmMemoryGain))
						helmMemoryGain = 0.0f;

					if (helmMemoryGain > 0.0f)
					{
						const float batchScale = static_cast<float>(tensorDff.batchCount) * helmMemoryGain;
						for (unsigned int j = 0; j < outputDim; ++j)
						{
							const float predictedResidual = clipf_maybe(predictedResidualW[j] * residualStd[j], gradClip);
							if (!is_finite(predictedResidual))
								continue;
							if (j < outTr.gBias.size())
								outTr.gBias[j] += batchScale * predictedResidual;
							const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(outTr.in);
							for (unsigned int i = 0; i < outputHiddenDim; ++i)
								outTr.gW[rowOff + i] += batchScale * predictedResidual * rawHiddenMean[outputHiddenOffset + i];
						}
					}

					for (unsigned int m = 0; m < modeRank; ++m)
					{
						if (m >= helm.poleNumer.size() || m >= helm.poleDenom.size() || m >= helm.pole.size() || m >= helm.latent.size())
							continue;
						const float oldLatent = helm.latent[m];
						const float poleNumer = (betaHelm * helm.poleNumer[m])
						                      + ((1.0f - betaHelm) * predictedLatent[m] * oldLatent);
						const float poleDenom = (betaHelm * helm.poleDenom[m])
						                      + ((1.0f - betaHelm) * oldLatent * oldLatent);
						helm.poleNumer[m] = poleNumer;
						helm.poleDenom[m] = poleDenom;
						if (poleDenom > sigmaEps)
							helm.pole[m] = std::max(-acHelm.helmPoleMax,
							                        std::min(acHelm.helmPoleMax, poleNumer / (poleDenom + sigmaEps)));
						helm.latent[m] = predictedLatent[m];
					}

					for (unsigned int j = 0; j < outputDim; ++j)
					{
						const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(pastDim);
						for (unsigned int p = 0; p < pastDim; ++p)
						{
							const float sample = currentResidualW[j] * pastSignal[p];
							helm.crossCov[rowOff + p] =
							    (betaHelm * helm.crossCov[rowOff + p]) + ((1.0f - betaHelm) * sample);
						}
					}

					if (hiddenDim > 0u && outputDim > 0u && pastDim > 0u)
					{
						for (unsigned int j1 = 0; j1 < outputDim; ++j1)
						{
							for (unsigned int j2 = 0; j2 < outputDim; ++j2)
							{
								double accum = 0.0;
								const size_t rowOff1 = static_cast<size_t>(j1) * static_cast<size_t>(pastDim);
								const size_t rowOff2 = static_cast<size_t>(j2) * static_cast<size_t>(pastDim);
								for (unsigned int p = 0; p < pastDim; ++p)
									accum += static_cast<double>(helm.crossCov[rowOff1 + p]) * static_cast<double>(helm.crossCov[rowOff2 + p]);
								gram[static_cast<size_t>(j1) * static_cast<size_t>(outputDim) + j2] = static_cast<float>(accum);
							}
						}

						for (unsigned int m = 0; m < modeRank; ++m)
						{
							std::vector<float> leftVec(outputDim, 0.0f);
							bool seeded = false;
							const size_t oldLeftOff = static_cast<size_t>(m) * static_cast<size_t>(outputDim);
							double leftNormSq = 0.0;
							for (unsigned int j = 0; j < outputDim && (oldLeftOff + j) < helm.leftMode.size(); ++j)
							{
								leftVec[j] = helm.leftMode[oldLeftOff + j];
								leftNormSq += static_cast<double>(leftVec[j]) * static_cast<double>(leftVec[j]);
							}
							if (leftNormSq > 1e-12)
							{
								const float invNorm = 1.0f / static_cast<float>(sqrt(leftNormSq));
								for (unsigned int j = 0; j < outputDim; ++j)
									leftVec[j] *= invNorm;
								seeded = true;
							}
							if (!seeded && m < outputDim)
							{
								leftVec[m] = 1.0f;
								seeded = true;
							}
							if (!seeded && outputDim > 0u)
							{
								leftVec[0] = 1.0f;
								seeded = true;
							}
							for (unsigned int iter = 0; iter < 6u && seeded; ++iter)
							{
								std::vector<float> nextVec(outputDim, 0.0f);
								for (unsigned int j = 0; j < outputDim; ++j)
								{
									double accum = 0.0;
									for (unsigned int k = 0; k < outputDim; ++k)
										accum += static_cast<double>(gram[static_cast<size_t>(j) * static_cast<size_t>(outputDim) + k])
										      * static_cast<double>(leftVec[k]);
									nextVec[j] = static_cast<float>(accum);
								}
								for (unsigned int pm = 0; pm < m; ++pm)
								{
									const size_t prevOff = static_cast<size_t>(pm) * static_cast<size_t>(outputDim);
									double proj = 0.0;
									for (unsigned int j = 0; j < outputDim; ++j)
										proj += static_cast<double>(nextVec[j]) * static_cast<double>(nextLeft[prevOff + j]);
									for (unsigned int j = 0; j < outputDim; ++j)
										nextVec[j] -= static_cast<float>(proj * static_cast<double>(nextLeft[prevOff + j]));
								}
								double normSq = 0.0;
								for (unsigned int j = 0; j < outputDim; ++j)
									normSq += static_cast<double>(nextVec[j]) * static_cast<double>(nextVec[j]);
								if (normSq <= 1e-18)
								{
									seeded = false;
									break;
								}
								const float invNorm = 1.0f / static_cast<float>(sqrt(normSq));
								for (unsigned int j = 0; j < outputDim; ++j)
									leftVec[j] = nextVec[j] * invNorm;
							}
							if (!seeded)
								continue;

							std::vector<float> gramLeft(outputDim, 0.0f);
							for (unsigned int j = 0; j < outputDim; ++j)
							{
								double accum = 0.0;
								for (unsigned int k = 0; k < outputDim; ++k)
									accum += static_cast<double>(gram[static_cast<size_t>(j) * static_cast<size_t>(outputDim) + k])
									      * static_cast<double>(leftVec[k]);
								gramLeft[j] = static_cast<float>(accum);
							}
							double lambda = 0.0;
							for (unsigned int j = 0; j < outputDim; ++j)
								lambda += static_cast<double>(leftVec[j]) * static_cast<double>(gramLeft[j]);
							const float sigma = static_cast<float>(sqrt(std::max(0.0, lambda)));
							if (!(sigma > sigmaEps))
								continue;

							const size_t leftOff = static_cast<size_t>(m) * static_cast<size_t>(outputDim);
							for (unsigned int j = 0; j < outputDim; ++j)
								nextLeft[leftOff + j] = leftVec[j];
							nextSigma[m] = sigma;

							std::vector<float> rightVec(pastDim, 0.0f);
							for (unsigned int p = 0; p < pastDim; ++p)
							{
								double accum = 0.0;
								for (unsigned int j = 0; j < outputDim; ++j)
								{
									const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(pastDim);
									accum += static_cast<double>(helm.crossCov[rowOff + p]) * static_cast<double>(leftVec[j]);
								}
								rightVec[p] = static_cast<float>(accum / sigma);
							}
							for (unsigned int pm = 0; pm < m; ++pm)
							{
								const size_t prevOff = static_cast<size_t>(pm) * static_cast<size_t>(pastDim);
								double proj = 0.0;
								for (unsigned int p = 0; p < pastDim; ++p)
									proj += static_cast<double>(rightVec[p]) * static_cast<double>(nextRight[prevOff + p]);
								for (unsigned int p = 0; p < pastDim; ++p)
									rightVec[p] -= static_cast<float>(proj * static_cast<double>(nextRight[prevOff + p]));
							}
							double rightNormSq = 0.0;
							for (unsigned int p = 0; p < pastDim; ++p)
								rightNormSq += static_cast<double>(rightVec[p]) * static_cast<double>(rightVec[p]);
							if (rightNormSq <= 1e-18)
							{
								nextSigma[m] = 0.0f;
								for (unsigned int j = 0; j < outputDim; ++j)
									nextLeft[leftOff + j] = 0.0f;
								continue;
							}
							const float invNorm = 1.0f / static_cast<float>(sqrt(rightNormSq));
							const size_t rightOff = static_cast<size_t>(m) * static_cast<size_t>(pastDim);
							for (unsigned int p = 0; p < pastDim; ++p)
								nextRight[rightOff + p] = rightVec[p] * invNorm;
						}
					}

					unsigned int helmActiveModes = 0u;
					double nextEffectiveSigmaSq = 0.0;
					for (unsigned int m = 0; m < modeRank; ++m)
					{
						if (m < helm.sigma.size())
							helm.sigma[m] = nextSigma[m];
						nextEffectiveSigmaSq += static_cast<double>(nextSigma[m]) * static_cast<double>(nextSigma[m]);
						if (nextSigma[m] > acHelm.helmEdgeThreshold)
							helmActiveModes += 1u;
					}
					helm.leftMode.swap(nextLeft);
					helm.rightMode.swap(nextRight);
					helm.lastActiveModes = helmActiveModes;
					helm.lastEdge = (modeRank > 0u) ? std::max(0.0f, helm.sigma[0]) : 0.0f;
					helm.lastSecondEdge = (modeRank > 1u) ? std::max(0.0f, helm.sigma[1]) : 0.0f;
					helm.lastSecondEdgeRatio = (helm.lastEdge > sigmaEps)
					                         ? std::max(0.0f, helm.lastSecondEdge / (helm.lastEdge + sigmaEps))
					                         : 0.0f;
					helm.lastSigma = static_cast<float>(sqrt(nextEffectiveSigmaSq));
					helm.lastPredR2 = helmPredR2;
					helm.lastMemoryGain = helmMemoryGain;
					for (unsigned int i = 0; i < hiddenDim; ++i)
						helm.prevHiddenMean[i] = hiddenMean[i];
					for (unsigned int j = 0; j < outputDim; ++j)
						helm.prevResidualMean[j] = residualMean[j];
				}

				if (asterEnabled && lastTransition < tensorDff.T.size())
				{
					const timespec asterBoundaryStart = monotonic_now();
					TensorDFFState::Transition& outTr = tensorDff.T[lastTransition];
					const ATLASConfig& acAster = trainingConfig.atlas;
					const unsigned int rawHiddenDim = aster.rawHiddenDim;
					const unsigned int controlDim = aster.controlDim;
					const unsigned int outputDim = std::min<unsigned int>(aster.outputDim, outTr.out);
					const unsigned int featureDim = outputDim + (2u * controlDim);
					const unsigned int stateFeatureDim =
					    std::max(1u, std::min<unsigned int>(aster.stateRank, std::max(1u, outputDim))) + (2u * controlDim);
					const unsigned int stateRank = std::min<unsigned int>(std::max(1u, aster.stateRank),
					                                                    std::max(1u, outputDim));
					const unsigned int trackedLayers = std::min<unsigned int>(
					    static_cast<unsigned int>(aster.hiddenLayerActivationIndices.size()),
					    std::min<unsigned int>(static_cast<unsigned int>(aster.hiddenLayerOffsets.size()),
					                           static_cast<unsigned int>(aster.hiddenLayerSizes.size())));
					const float varEps = 1e-6f;
					const float sigmaEps = 1e-12f;
					const float betaAster = std::min<float>(std::max<float>(acAster.beta, 0.0f), 1.0f);
					const float ridge = 1e-4f;
					std::vector<float> rawHiddenMean(rawHiddenDim, 0.0f);
					std::vector<float> controlMean(controlDim, 0.0f);
					std::vector<float> residualMean(outputDim, 0.0f);
					std::vector<float> controlStd(controlDim, 1.0f);
					std::vector<float> residualStd(outputDim, 1.0f);
					std::vector<float> feature(featureDim, 0.0f);
					std::vector<float> prevControlW(controlDim, 0.0f);
					std::vector<float> currentControlW(controlDim, 0.0f);
					std::vector<float> currentResidualW(outputDim, 0.0f);
					std::vector<float> predictedResidualCurrentW(outputDim, 0.0f);
					std::vector<float> predictedResidualNextW(outputDim, 0.0f);
					std::vector<float> observedLatent(stateRank, 0.0f);
					std::vector<float> prevLatentAligned(stateRank, 0.0f);
					std::vector<float> predictedState(stateRank, 0.0f);
					std::vector<float> filteredState(stateRank, 0.0f);
					std::vector<float> nextState(stateRank, 0.0f);
					std::vector<float> innovation(outputDim, 0.0f);
					std::vector<float> stateCorrection(stateRank, 0.0f);
					std::vector<float> stateFeature(stateFeatureDim, 0.0f);
					std::vector<float> invPastCov;
					std::vector<float> invStatePastCov;
					std::vector<float> invInnovationCov;
					std::vector<float> theta(static_cast<size_t>(outputDim) * static_cast<size_t>(featureDim), 0.0f);
					std::vector<float> stateModel(static_cast<size_t>(stateRank) * static_cast<size_t>(stateFeatureDim), 0.0f);
					std::vector<float> innovationGain(static_cast<size_t>(stateRank) * static_cast<size_t>(outputDim), 0.0f);
					std::vector<float> nextSigma;
					std::vector<float> nextLeft;
					std::vector<float> nextRight;
					const unsigned int outputHiddenOffset =
					    aster.hiddenLayerOffsets.empty() ? 0u : aster.hiddenLayerOffsets.back();
					const unsigned int outputHiddenDim =
					    (aster.hiddenLayerSizes.empty() || outputHiddenOffset >= rawHiddenDim)
					        ? 0u
					        : std::min<unsigned int>(
					              std::min<unsigned int>(aster.hiddenLayerSizes.back(), outTr.in),
					              rawHiddenDim - outputHiddenOffset);
					const double asterSetupNs =
					    monotonic_elapsed_ns(asterBoundaryStart, monotonic_now());

					for (unsigned int i = 0; i < rawHiddenDim; ++i)
						rawHiddenMean[i] = aster.batchHiddenSum[i] * invBatch;

					const timespec asterTransportStart = monotonic_now();
					for (unsigned int l = 0; l < trackedLayers && ((l + 1u) * outputDim) <= controlDim; ++l)
					{
						const unsigned int layerOffset = aster.hiddenLayerOffsets[l];
						const unsigned int layerSize = (layerOffset < rawHiddenDim)
						                             ? std::min<unsigned int>(aster.hiddenLayerSizes[l], rawHiddenDim - layerOffset)
						                             : 0u;
						if (layerSize == 0u)
							continue;

						std::vector<float> transported(layerSize, 0.0f);
						for (unsigned int i = 0; i < layerSize; ++i)
							transported[i] = rawHiddenMean[layerOffset + i];
						unsigned int currentActIndex = aster.hiddenLayerActivationIndices[l];

						for (unsigned int dl = l + 1u; dl < trackedLayers; ++dl)
						{
							if (currentActIndex >= tensorDff.T.size())
							{
								transported.clear();
								break;
							}
							const TensorDFFState::Transition& downTr = tensorDff.T[currentActIndex];
							const unsigned int nextOffset = aster.hiddenLayerOffsets[dl];
							const unsigned int nextSize = (nextOffset < rawHiddenDim)
							                            ? std::min<unsigned int>(aster.hiddenLayerSizes[dl], rawHiddenDim - nextOffset)
							                            : 0u;
							if (nextSize == 0u || transported.empty())
							{
								transported.clear();
								break;
							}

							std::vector<float> nextTransport(nextSize, 0.0f);
							const unsigned int dotIn = std::min<unsigned int>(static_cast<unsigned int>(transported.size()), downTr.in);
							const unsigned int dotOut = std::min<unsigned int>(nextSize, downTr.out);
							const int nextActFx = skeleton->getActivationType(static_cast<int>(currentActIndex));
							const float nextActParam = skeleton->getActivationParam(static_cast<int>(currentActIndex));
							for (unsigned int j = 0; j < dotOut; ++j)
							{
								double accum = 0.0;
								const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(downTr.in);
								for (unsigned int i = 0; i < dotIn; ++i)
									accum += static_cast<double>(downTr.W[rowOff + i]) * static_cast<double>(transported[i]);
								const float nextActivation = rawHiddenMean[nextOffset + j];
								const float gate = GMath::activationErrDer(nextActivation, nextActFx, nextActParam);
								nextTransport[j] = static_cast<float>(accum) * gate;
							}
							transported.swap(nextTransport);
							currentActIndex = aster.hiddenLayerActivationIndices[dl];
						}

						if (transported.empty())
							continue;

						const unsigned int projectedOff = l * outputDim;
						const unsigned int dotIn = std::min<unsigned int>(static_cast<unsigned int>(transported.size()), outTr.in);
						for (unsigned int j = 0; j < outputDim; ++j)
						{
							double accum = 0.0;
							const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(outTr.in);
							for (unsigned int i = 0; i < dotIn; ++i)
								accum += static_cast<double>(outTr.W[rowOff + i]) * static_cast<double>(transported[i]);
							controlMean[projectedOff + j] = static_cast<float>(accum);
						}
					}
					const double asterTransportNs =
					    monotonic_elapsed_ns(asterTransportStart, monotonic_now());

					const timespec asterTransferFitStart = monotonic_now();
					for (unsigned int j = 0; j < outputDim; ++j)
					{
						residualMean[j] = aster.batchResidualSum[j] * invBatch;
						const float secondMoment = aster.batchResidualSqSum[j] * invBatch;
						const float updatedVar =
						    (betaAster * aster.residualVar[j]) + ((1.0f - betaAster) * std::max(secondMoment, varEps));
						aster.residualVar[j] = std::max(updatedVar, varEps);
						residualStd[j] = sqrtf(aster.residualVar[j]);
						currentResidualW[j] = (residualStd[j] > sigmaEps) ? (residualMean[j] / residualStd[j]) : 0.0f;
						feature[j] = (residualStd[j] > sigmaEps) ? (aster.prevResidualMean[j] / residualStd[j]) : 0.0f;
					}
					for (unsigned int i = 0; i < controlDim; ++i)
					{
						const float secondMoment = controlMean[i] * controlMean[i];
						const float updatedVar =
						    (betaAster * aster.controlVar[i]) + ((1.0f - betaAster) * std::max(secondMoment, varEps));
						aster.controlVar[i] = std::max(updatedVar, varEps);
						controlStd[i] = sqrtf(aster.controlVar[i]);
						prevControlW[i] =
						    (controlStd[i] > sigmaEps) ? (aster.prevControlMean[i] / controlStd[i]) : 0.0f;
						currentControlW[i] =
						    (controlStd[i] > sigmaEps) ? (controlMean[i] / controlStd[i]) : 0.0f;
						feature[outputDim + i] = prevControlW[i];
						feature[outputDim + controlDim + i] = currentControlW[i];
					}

					for (unsigned int r = 0; r < featureDim; ++r)
					{
						const size_t rowOff = static_cast<size_t>(r) * static_cast<size_t>(featureDim);
						for (unsigned int c = 0; c < featureDim; ++c)
						{
							const float sample = feature[r] * feature[c];
							aster.pastCov[rowOff + c] =
							    (betaAster * aster.pastCov[rowOff + c]) + ((1.0f - betaAster) * sample);
						}
					}
					for (unsigned int j = 0; j < outputDim; ++j)
					{
						const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(featureDim);
						for (unsigned int c = 0; c < featureDim; ++c)
						{
							const float sample = currentResidualW[j] * feature[c];
							aster.crossCov[rowOff + c] =
							    (betaAster * aster.crossCov[rowOff + c]) + ((1.0f - betaAster) * sample);
						}
					}

					if (invert_small_dense_row_major(aster.pastCov, featureDim, ridge, invPastCov))
					{
						for (unsigned int j = 0; j < outputDim; ++j)
						{
							const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(featureDim);
							for (unsigned int c = 0; c < featureDim; ++c)
							{
								double accum = 0.0;
								for (unsigned int k = 0; k < featureDim; ++k)
								{
									accum += static_cast<double>(aster.crossCov[rowOff + k])
									      * static_cast<double>(invPastCov[static_cast<size_t>(k) * static_cast<size_t>(featureDim) + c]);
								}
								theta[rowOff + c] = static_cast<float>(accum);
							}
						}
					}

					aster.theta = theta;
					extract_top_singular_modes_row_major(theta, outputDim, featureDim, stateRank,
					                                     nextSigma, nextLeft, nextRight);

					for (unsigned int m = 0; m < stateRank; ++m)
					{
						bool aligned = false;
						if ((m < aster.latent.size())
						    && (aster.leftMode.size() >= static_cast<size_t>(stateRank) * static_cast<size_t>(outputDim))
						    && (nextLeft.size() >= static_cast<size_t>(stateRank) * static_cast<size_t>(outputDim)))
						{
							double accum = 0.0;
							const size_t nextLeftOff = static_cast<size_t>(m) * static_cast<size_t>(outputDim);
							for (unsigned int pm = 0; pm < stateRank && pm < aster.latent.size(); ++pm)
							{
								const size_t prevLeftOff = static_cast<size_t>(pm) * static_cast<size_t>(outputDim);
								double corr = 0.0;
								for (unsigned int j = 0; j < outputDim; ++j)
								{
									corr += static_cast<double>(nextLeft[nextLeftOff + j])
									      * static_cast<double>(aster.leftMode[prevLeftOff + j]);
								}
								accum += corr * static_cast<double>(aster.latent[pm]);
							}
							prevLatentAligned[m] = static_cast<float>(accum);
							aligned = true;
						}
						if (!aligned && m < aster.latent.size())
							prevLatentAligned[m] = aster.latent[m];
					}

					for (unsigned int m = 0; m < stateRank; ++m)
						stateFeature[m] = prevLatentAligned[m];
					for (unsigned int i = 0; i < controlDim; ++i)
					{
						stateFeature[stateRank + i] = prevControlW[i];
						stateFeature[stateRank + controlDim + i] = currentControlW[i];
					}

					for (unsigned int m = 0; m < stateRank; ++m)
					{
						const size_t leftOff = static_cast<size_t>(m) * static_cast<size_t>(outputDim);
						for (unsigned int j = 0; j < outputDim && (leftOff + j) < nextLeft.size(); ++j)
							observedLatent[m] += nextLeft[leftOff + j] * currentResidualW[j];
					}
					const double asterTransferFitNs =
					    monotonic_elapsed_ns(asterTransferFitStart, monotonic_now());

					const timespec asterStateFitStart = monotonic_now();
					if (aster.statePastCov.size() >= static_cast<size_t>(stateFeatureDim) * static_cast<size_t>(stateFeatureDim)
					    && aster.stateCrossCov.size() >= static_cast<size_t>(stateRank) * static_cast<size_t>(stateFeatureDim))
					{
						for (unsigned int r = 0; r < stateFeatureDim; ++r)
						{
							const size_t rowOff = static_cast<size_t>(r) * static_cast<size_t>(stateFeatureDim);
							for (unsigned int c = 0; c < stateFeatureDim; ++c)
							{
								const float sample = stateFeature[r] * stateFeature[c];
								aster.statePastCov[rowOff + c] =
								    (betaAster * aster.statePastCov[rowOff + c]) + ((1.0f - betaAster) * sample);
							}
						}
						for (unsigned int m = 0; m < stateRank; ++m)
						{
							const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(stateFeatureDim);
							for (unsigned int c = 0; c < stateFeatureDim; ++c)
							{
								const float sample = observedLatent[m] * stateFeature[c];
								aster.stateCrossCov[rowOff + c] =
								    (betaAster * aster.stateCrossCov[rowOff + c]) + ((1.0f - betaAster) * sample);
							}
						}
						if (invert_small_dense_row_major(aster.statePastCov, stateFeatureDim, ridge, invStatePastCov))
						{
							for (unsigned int m = 0; m < stateRank; ++m)
							{
								const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(stateFeatureDim);
								for (unsigned int c = 0; c < stateFeatureDim; ++c)
								{
									double accum = 0.0;
									for (unsigned int k = 0; k < stateFeatureDim; ++k)
									{
										accum += static_cast<double>(aster.stateCrossCov[rowOff + k])
										      * static_cast<double>(invStatePastCov[static_cast<size_t>(k) * static_cast<size_t>(stateFeatureDim) + c]);
									}
									stateModel[rowOff + c] = static_cast<float>(accum);
								}
							}
						}
					}

					float maxRowAbs = 0.0f;
					for (unsigned int m = 0; m < stateRank; ++m)
					{
						const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(stateFeatureDim);
						float rowAbs = 0.0f;
						for (unsigned int s = 0; s < stateRank; ++s)
							rowAbs += std::fabs(stateModel[rowOff + s]);
						maxRowAbs = std::max(maxRowAbs, rowAbs);
					}
					if (maxRowAbs > acAster.asterPoleMax && maxRowAbs > sigmaEps)
					{
						const float scale = acAster.asterPoleMax / maxRowAbs;
						for (unsigned int m = 0; m < stateRank; ++m)
						{
							const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(stateFeatureDim);
							for (unsigned int s = 0; s < stateRank; ++s)
								stateModel[rowOff + s] *= scale;
						}
					}

					for (unsigned int m = 0; m < stateRank; ++m)
					{
						const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(stateFeatureDim);
						double accum = 0.0;
						for (unsigned int s = 0; s < stateRank; ++s)
							accum += static_cast<double>(stateModel[rowOff + s]) * static_cast<double>(prevLatentAligned[s]);
						for (unsigned int i = 0; i < controlDim; ++i)
						{
							accum += static_cast<double>(stateModel[rowOff + stateRank + i]) * static_cast<double>(prevControlW[i]);
							accum += static_cast<double>(stateModel[rowOff + stateRank + controlDim + i]) * static_cast<double>(currentControlW[i]);
						}
						predictedState[m] = static_cast<float>(accum);
					}
					for (unsigned int m = 0; m < stateRank; ++m)
					{
						const size_t leftOff = static_cast<size_t>(m) * static_cast<size_t>(outputDim);
						for (unsigned int j = 0; j < outputDim && (leftOff + j) < nextLeft.size(); ++j)
							predictedResidualCurrentW[j] += nextLeft[leftOff + j] * predictedState[m];
					}

					double targetNormSq = 0.0;
					double errNormSq = 0.0;
					for (unsigned int j = 0; j < outputDim; ++j)
					{
						const double target = static_cast<double>(currentResidualW[j]);
						const double err = target - static_cast<double>(predictedResidualCurrentW[j]);
						innovation[j] = static_cast<float>(err);
						targetNormSq += target * target;
						errNormSq += err * err;
					}
					float asterPredR2 = 0.0f;
					if (targetNormSq > 1e-12)
						asterPredR2 = static_cast<float>(std::max<double>(0.0, 1.0 - (errNormSq / targetNormSq)));
					const double asterStateFitNs =
					    monotonic_elapsed_ns(asterStateFitStart, monotonic_now());

					for (unsigned int m = 0; m < stateRank; ++m)
						stateCorrection[m] = observedLatent[m] - predictedState[m];
					const timespec asterInnovationFitStart = monotonic_now();
					if (aster.innovationCov.size() >= static_cast<size_t>(outputDim) * static_cast<size_t>(outputDim)
					    && aster.innovationCross.size() >= static_cast<size_t>(stateRank) * static_cast<size_t>(outputDim))
					{
						for (unsigned int r = 0; r < outputDim; ++r)
						{
							const size_t rowOff = static_cast<size_t>(r) * static_cast<size_t>(outputDim);
							for (unsigned int c = 0; c < outputDim; ++c)
							{
								const float sample = innovation[r] * innovation[c];
								aster.innovationCov[rowOff + c] =
								    (betaAster * aster.innovationCov[rowOff + c]) + ((1.0f - betaAster) * sample);
							}
						}
						for (unsigned int m = 0; m < stateRank; ++m)
						{
							const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(outputDim);
							for (unsigned int j = 0; j < outputDim; ++j)
							{
								const float sample = stateCorrection[m] * innovation[j];
								aster.innovationCross[rowOff + j] =
								    (betaAster * aster.innovationCross[rowOff + j]) + ((1.0f - betaAster) * sample);
							}
						}
						if (invert_small_dense_row_major(aster.innovationCov, outputDim, ridge, invInnovationCov))
						{
							for (unsigned int m = 0; m < stateRank; ++m)
							{
								const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(outputDim);
								for (unsigned int j = 0; j < outputDim; ++j)
								{
									double accum = 0.0;
									for (unsigned int k = 0; k < outputDim; ++k)
									{
										accum += static_cast<double>(aster.innovationCross[rowOff + k])
										      * static_cast<double>(invInnovationCov[static_cast<size_t>(k) * static_cast<size_t>(outputDim) + j]);
									}
									innovationGain[rowOff + j] = static_cast<float>(accum);
								}
							}
						}
					}

					for (unsigned int m = 0; m < stateRank; ++m)
					{
						double accum = static_cast<double>(predictedState[m]);
						const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(outputDim);
						for (unsigned int j = 0; j < outputDim; ++j)
							accum += static_cast<double>(innovationGain[rowOff + j]) * static_cast<double>(innovation[j]);
						filteredState[m] = static_cast<float>(accum);
					}
					for (unsigned int m = 0; m < stateRank; ++m)
					{
						const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(stateFeatureDim);
						double accum = 0.0;
						for (unsigned int s = 0; s < stateRank; ++s)
							accum += static_cast<double>(stateModel[rowOff + s]) * static_cast<double>(filteredState[s]);
						for (unsigned int i = 0; i < controlDim; ++i)
						{
							accum += static_cast<double>(stateModel[rowOff + stateRank + i]) * static_cast<double>(currentControlW[i]);
							accum += static_cast<double>(stateModel[rowOff + stateRank + controlDim + i]) * static_cast<double>(currentControlW[i]);
						}
						nextState[m] = static_cast<float>(accum);
					}
					for (unsigned int m = 0; m < stateRank; ++m)
					{
						const size_t leftOff = static_cast<size_t>(m) * static_cast<size_t>(outputDim);
						for (unsigned int j = 0; j < outputDim && (leftOff + j) < nextLeft.size(); ++j)
							predictedResidualNextW[j] += nextLeft[leftOff + j] * nextState[m];
					}
					const double asterInnovationFitNs =
					    monotonic_elapsed_ns(asterInnovationFitStart, monotonic_now());

					const timespec asterApplyStart = monotonic_now();
					const float asterEdge = (!nextSigma.empty()) ? std::max(0.0f, nextSigma[0]) : 0.0f;
					float asterTransferScale = 0.0f;
					if (asterEdge > acAster.asterEdgeThreshold)
						asterTransferScale = (asterEdge - acAster.asterEdgeThreshold) / (asterEdge + 1e-6f);
					const float outputEvidenceRaw =
					    std::max(0.0f, asterEdge) * std::max(0.0f, asterPredR2);
					float predictiveEvidence = 0.0f;
					float meritGeometryTrust = 0.0f;
					if (lastTransition < tensorDff.atlasState.size())
					{
						const atlas::WeightState& outputAtlas = tensorDff.atlasState[lastTransition];
						predictiveEvidence =
						    std::max(0.0f, outputAtlas.lastSparrowEdge)
						    * std::max(0.0f, outputAtlas.lastSparrowHorizontalRatio);
						if (outputAtlas.activeRank > 0u
						    && !outputAtlas.fisherDiag.empty()
						    && outputAtlas.totalTrace > 1.0e-6f)
						{
							const unsigned int captureRank =
							    std::min(outputAtlas.activeRank,
							             static_cast<unsigned int>(outputAtlas.fisherDiag.size()));
							double activeTrace = 0.0;
							for (unsigned int c = 0u; c < captureRank; ++c)
								activeTrace += static_cast<double>(outputAtlas.fisherDiag[c]);
							const float capture =
							    static_cast<float>(activeTrace / (static_cast<double>(outputAtlas.totalTrace) + 1.0e-6));
							meritGeometryTrust =
							    std::max(0.0f,
							             std::min(1.0f, acAster.meritGeometryScale * std::max(0.0f, capture)));
						}
					}
					float asterMemoryGain = acAster.asterMemoryScale
					                      * std::max(0.0f, asterTransferScale)
					                      * std::max(0.0f, asterPredR2);
					if (!is_finite(asterMemoryGain))
						asterMemoryGain = 0.0f;
					float aegisLambdaSpatial = 0.0f;
					float aegisLambdaPredictive = 0.0f;
					float aegisLambdaOutput = 0.0f;
					float aegisPredictivePredicted = 0.0f;
					float aegisPredictiveRealized = predictiveEvidence;
					float aegisOutputPredicted = 0.0f;
					float aegisOutputRealized = outputEvidenceRaw;
					float citadelAnchor = 0.0f;
					float citadelHardRegimeMass = 0.0f;
					float citadelSparrowTrust = 1.0f;
					float rampartTau = 0.0f;
					float rampartBudget = 0.0f;
					float rampartCovariance = 0.0f;
					float rampartSparrowTrust = 1.0f;
					float meritTau = 0.0f;
					float meritBudget = 0.0f;
					float meritCovariance = 0.0f;
					float meritSparrowTrust = 1.0f;
					float strataNullMode = 1.0f;
					float strataPredictiveMode = 0.0f;
					float strataOutputMode = 0.0f;
					float strataCoupledMode = 0.0f;
					float strataBudget = 0.0f;
					float strataNullBenefit = 0.0f;
					float strataPredictiveBenefit = 0.0f;
					float strataOutputBenefit = 0.0f;
					float strataCoupledBenefit = 0.0f;
					float strataSelectedExcess = 0.0f;
					float strataSwitchRate = 0.0f;
					if (acAster.aegisEnabled || acAster.auroraEnabled || acAster.seamEnabled || acAster.quasarEnabled)
					{
						const float aegisCalibBeta = 0.90f;
						const float aegisPrecisionFloor = 0.025f;
						const float aegisPrecisionMax = 32.0f;
						const float citadelTrustBeta = 0.90f;
						const float strataBenefitBeta = 0.90f;
						aegisPredictivePredicted = std::max(0.0f, aster.aegisPrevPredictiveScore);
						aegisOutputPredicted = std::max(0.0f, aster.aegisPrevOutputScore);
						if (aster.timingBoundaryCount > 0ULL)
						{
							const float predictiveErr =
							    fabsf(aegisPredictivePredicted - aegisPredictiveRealized);
							const float outputErr =
							    fabsf(aegisOutputPredicted - aegisOutputRealized);
							aster.aegisPredictiveErrorEma =
							    aegisCalibBeta * aster.aegisPredictiveErrorEma
							    + (1.0f - aegisCalibBeta) * predictiveErr;
							aster.aegisOutputErrorEma =
							    aegisCalibBeta * aster.aegisOutputErrorEma
							    + (1.0f - aegisCalibBeta) * outputErr;
						}
						const float rawPredictiveTrust =
						    std::min(aegisPrecisionMax,
						             acAster.aegisPredictiveScale * std::max(0.0f, predictiveEvidence)
						                 / (aster.aegisPredictiveErrorEma + aegisPrecisionFloor));
						const float rawOutputTrust =
						    std::min(aegisPrecisionMax,
						             acAster.aegisOutputScale * outputEvidenceRaw
						                 / (aster.aegisOutputErrorEma + aegisPrecisionFloor));
						if (aster.timingBoundaryCount > 0ULL)
						{
							aster.citadelPredictiveTrustEma =
							    citadelTrustBeta * aster.citadelPredictiveTrustEma
							    + (1.0f - citadelTrustBeta) * rawPredictiveTrust;
							aster.citadelOutputTrustEma =
							    citadelTrustBeta * aster.citadelOutputTrustEma
							    + (1.0f - citadelTrustBeta) * rawOutputTrust;
						}
						else
						{
							aster.citadelPredictiveTrustEma = rawPredictiveTrust;
							aster.citadelOutputTrustEma = rawOutputTrust;
						}
						const float disagreementNorm =
						    fabsf(aegisPredictiveRealized - aegisOutputRealized)
						    / (fabsf(aegisPredictiveRealized) + fabsf(aegisOutputRealized) + 1e-6f);
						float residualScale = 1.0f;
						float spatialPrecision = 1.0f;
						if (acAster.auroraEnabled)
						{
							const float predictiveTrustNorm =
							    rawPredictiveTrust / (rawPredictiveTrust + 1.0f);
							const float outputTrustNorm =
							    rawOutputTrust / (rawOutputTrust + 1.0f);
							const float predictiveErrNorm =
							    std::max(0.0f,
							             std::min(1.0f,
							                      aster.aegisPredictiveErrorEma
							                          / (aegisPredictivePredicted
							                             + aegisPredictiveRealized + 1.0e-6f)));
							const float outputErrNorm =
							    std::max(0.0f,
							             std::min(1.0f,
							                      aster.aegisOutputErrorEma
							                          / (aegisOutputPredicted
							                             + aegisOutputRealized + 1.0e-6f)));
							const float uncertainty = 0.5f * (predictiveErrNorm + outputErrNorm);
							const float horizonPred =
							    acAster.auroraHorizonBlend * aster.citadelPredictiveTrustEma
							    + (1.0f - acAster.auroraHorizonBlend) * rawPredictiveTrust;
							const float horizonOut =
							    acAster.auroraHorizonBlend * aster.citadelOutputTrustEma
							    + (1.0f - acAster.auroraHorizonBlend) * rawOutputTrust;
							const float horizonPredNorm = horizonPred / (horizonPred + 1.0f);
							const float horizonOutNorm = horizonOut / (horizonOut + 1.0f);
							rampartTau =
							    0.35f + 1.65f
							                * std::max(0.0f,
							                           std::min(1.0f,
							                                    0.45f * uncertainty
							                                        + 0.35f * disagreementNorm));
							rampartCovariance =
							    std::max(0.0f,
							             std::min(0.95f,
							                      0.45f * std::min(horizonPredNorm, horizonOutNorm)
							                          * std::max(0.0f, 1.0f - disagreementNorm)
							                          * std::max(0.0f, 1.0f - 0.5f * uncertainty)));
							const float coupling =
							    rampartCovariance * sqrtf(std::max(0.0f, horizonPred * horizonOut));
							const float h11 = std::max(1.0e-6f, rampartTau + horizonPred);
							const float h22 = std::max(1.0e-6f, rampartTau + horizonOut);
							const float det = std::max(1.0e-6f, h11 * h22 - coupling * coupling);
							float wPredictive =
							    (horizonPred * horizonPredNorm * h22
							     - coupling * horizonOut * horizonOutNorm)
							    / det;
							float wOutput =
							    (h11 * horizonOut * horizonOutNorm
							     - coupling * horizonPred * horizonPredNorm)
							    / det;
							wPredictive = std::max(0.0f, wPredictive);
							wOutput = std::max(0.0f, wOutput);
							const float weightSum = wPredictive + wOutput + 1.0e-6f;
							wPredictive /= weightSum;
							wOutput /= weightSum;
							const float budgetSignal =
							    std::max(0.0f,
							             std::min(1.0f,
							                      0.55f * (0.5f * (horizonPredNorm + horizonOutNorm))
							                          + 0.20f * std::max(0.0f, 1.0f - uncertainty)
							                          - 0.20f * disagreementNorm));
							const float budgetTarget =
							    std::max(0.0f,
							             std::min(1.0f, acAster.auroraBudgetMax * budgetSignal));
							rampartBudget =
							    (aster.timingBoundaryCount > 0ULL)
							        ? (citadelTrustBeta * aster.rampartLastBudget
							           + (1.0f - citadelTrustBeta) * budgetTarget)
							        : budgetTarget;
							const float residualNorm =
							    sqrtf(std::max(0.0f,
							                   wPredictive * wPredictive + wOutput * wOutput
							                       + 2.0f * rampartCovariance * wPredictive * wOutput));
							const float budgetScale =
							    (residualNorm > 1.0e-6f)
							        ? std::max(0.0f, std::min(1.0f, rampartBudget / residualNorm))
							        : 0.0f;
							aegisLambdaPredictive =
							    std::max(0.0f, std::min(1.0f, wPredictive * budgetScale));
							aegisLambdaOutput =
							    std::max(0.0f, std::min(1.0f, wOutput * budgetScale));
							aegisLambdaSpatial =
							    std::max(0.0f, std::min(1.0f, 1.0f - aegisLambdaPredictive - aegisLambdaOutput));
							rampartSparrowTrust = aegisLambdaPredictive;
							asterMemoryGain *= std::max(0.0f, std::min(1.0f, aegisLambdaOutput));
						}
						else if (acAster.seamEnabled)
						{
							const float predictiveTrustNorm =
							    rawPredictiveTrust / (rawPredictiveTrust + 1.0f);
							const float outputTrustNorm =
							    rawOutputTrust / (rawOutputTrust + 1.0f);
							const float predictiveErrNorm =
							    std::max(0.0f,
							             std::min(1.0f,
							                      aster.aegisPredictiveErrorEma
							                          / (aegisPredictivePredicted
							                             + aegisPredictiveRealized + 1.0e-6f)));
							const float outputErrNorm =
							    std::max(0.0f,
							             std::min(1.0f,
							                      aster.aegisOutputErrorEma
							                          / (aegisOutputPredicted
							                             + aegisOutputRealized + 1.0e-6f)));
							const float uncertainty = 0.5f * (predictiveErrNorm + outputErrNorm);
							const float spatialTarget =
							    meritGeometryTrust - 0.35f * (predictiveTrustNorm + outputTrustNorm);
							const float predictiveTarget =
							    predictiveTrustNorm * std::max(0.0f, 1.0f - predictiveErrNorm)
							    - 0.20f * disagreementNorm;
							const float outputTarget =
							    outputTrustNorm * std::max(0.0f, 1.0f - outputErrNorm)
							    + 0.10f * std::max(0.0f, 1.0f - meritGeometryTrust)
							    - 0.15f * disagreementNorm;
							if (aster.timingBoundaryCount > 0ULL)
							{
								aster.strataNullBenefitEma =
								    0.85f * aster.strataNullBenefitEma
								    + acAster.seamMirrorStep * spatialTarget;
								aster.strataPredictiveBenefitEma =
								    0.85f * aster.strataPredictiveBenefitEma
								    + acAster.seamMirrorStep * predictiveTarget;
								aster.strataOutputBenefitEma =
								    0.85f * aster.strataOutputBenefitEma
								    + acAster.seamMirrorStep * outputTarget;
							}
							else
							{
								aster.strataNullBenefitEma = acAster.seamMirrorStep * spatialTarget;
								aster.strataPredictiveBenefitEma = acAster.seamMirrorStep * predictiveTarget;
								aster.strataOutputBenefitEma = acAster.seamMirrorStep * outputTarget;
							}
							const float maxLogit =
							    std::max(aster.strataNullBenefitEma,
							             std::max(aster.strataPredictiveBenefitEma,
							                      aster.strataOutputBenefitEma));
							const float expSpatial = expf(aster.strataNullBenefitEma - maxLogit);
							const float expPredictive = expf(aster.strataPredictiveBenefitEma - maxLogit);
							const float expOutput = expf(aster.strataOutputBenefitEma - maxLogit);
							const float coordSum = expSpatial + expPredictive + expOutput + 1.0e-6f;
							const float wSpatial = expSpatial / coordSum;
							const float wPredictive = expPredictive / coordSum;
							const float wOutput = expOutput / coordSum;
							const float residualMass =
							    std::max(0.0f, std::min(1.0f, 1.0f - wSpatial));
							meritBudget =
							    std::max(0.0f,
							             std::min(1.0f,
							                      acAster.seamBudgetMax * residualMass
							                          * std::max(0.0f,
							                                     1.0f - 0.5f * uncertainty
							                                         - 0.20f * disagreementNorm)));
							const float residualShare = wPredictive + wOutput + 1.0e-6f;
							aegisLambdaPredictive =
							    std::max(0.0f, std::min(1.0f, meritBudget * (wPredictive / residualShare)));
							aegisLambdaOutput =
							    std::max(0.0f, std::min(1.0f, meritBudget * (wOutput / residualShare)));
							aegisLambdaSpatial =
							    std::max(0.0f, std::min(1.0f, 1.0f - aegisLambdaPredictive - aegisLambdaOutput));
							meritSparrowTrust = aegisLambdaPredictive;
							asterMemoryGain *= std::max(0.0f, std::min(1.0f, aegisLambdaOutput));
						}
						else if (acAster.quasarEnabled)
						{
							const float predictiveTrustNorm =
							    rawPredictiveTrust / (rawPredictiveTrust + 1.0f);
							const float outputTrustNorm =
							    rawOutputTrust / (rawOutputTrust + 1.0f);
							const float predictiveErrNorm =
							    std::max(0.0f,
							             std::min(1.0f,
							                      aster.aegisPredictiveErrorEma
							                          / (aegisPredictivePredicted
							                             + aegisPredictiveRealized + 1.0e-6f)));
							const float outputErrNorm =
							    std::max(0.0f,
							             std::min(1.0f,
							                      aster.aegisOutputErrorEma
							                          / (aegisOutputPredicted
							                             + aegisOutputRealized + 1.0e-6f)));
							const float uncertainty = 0.5f * (predictiveErrNorm + outputErrNorm);
							const float residualPressure =
							    std::max(predictiveTrustNorm, outputTrustNorm);
							const float nullSignal =
							    0.20f + 0.45f * uncertainty + 0.30f * disagreementNorm - 0.25f * residualPressure;
							const float predictiveSignal =
							    predictiveTrustNorm * std::max(0.0f, 1.0f - predictiveErrNorm)
							    + 0.20f * meritGeometryTrust
							    - 0.10f * uncertainty - 0.15f * disagreementNorm;
							const float outputSignal =
							    outputTrustNorm * std::max(0.0f, 1.0f - outputErrNorm)
							    + 0.10f * std::max(0.0f, 1.0f - meritGeometryTrust)
							    - 0.10f * uncertainty - 0.10f * disagreementNorm;
							const float coupledSignal =
							    0.45f * (predictiveTrustNorm + outputTrustNorm)
							    + 0.30f * std::min(predictiveTrustNorm, outputTrustNorm)
							    + 0.15f * meritGeometryTrust
							    - 0.15f * uncertainty - 0.25f * disagreementNorm;
							if (aster.timingBoundaryCount > 0ULL)
							{
								aster.strataNullBenefitEma =
								    strataBenefitBeta * aster.strataNullBenefitEma
								    + (1.0f - strataBenefitBeta) * nullSignal;
								aster.strataPredictiveBenefitEma =
								    strataBenefitBeta * aster.strataPredictiveBenefitEma
								    + (1.0f - strataBenefitBeta) * predictiveSignal;
								aster.strataOutputBenefitEma =
								    strataBenefitBeta * aster.strataOutputBenefitEma
								    + (1.0f - strataBenefitBeta) * outputSignal;
								aster.strataCoupledBenefitEma =
								    strataBenefitBeta * aster.strataCoupledBenefitEma
								    + (1.0f - strataBenefitBeta) * coupledSignal;
							}
							else
							{
								aster.strataNullBenefitEma = nullSignal;
								aster.strataPredictiveBenefitEma = predictiveSignal;
								aster.strataOutputBenefitEma = outputSignal;
								aster.strataCoupledBenefitEma = coupledSignal;
							}
							const float temperature = std::max(0.05f, acAster.quasarTemperature);
							const float maxScore =
							    std::max(std::max(aster.strataNullBenefitEma, aster.strataPredictiveBenefitEma),
							             std::max(aster.strataOutputBenefitEma, aster.strataCoupledBenefitEma));
							const float qNull = expf((aster.strataNullBenefitEma - maxScore) / temperature);
							const float qPredictive = expf((aster.strataPredictiveBenefitEma - maxScore) / temperature);
							const float qOutput = expf((aster.strataOutputBenefitEma - maxScore) / temperature);
							const float qCoupled = expf((aster.strataCoupledBenefitEma - maxScore) / temperature);
							const float probSum = qNull + qPredictive + qOutput + qCoupled + 1.0e-6f;
							strataNullMode = qNull / probSum;
							strataPredictiveMode = qPredictive / probSum;
							strataOutputMode = qOutput / probSum;
							strataCoupledMode = qCoupled / probSum;
							strataBudget =
							    std::max(0.0f,
							             std::min(1.0f,
							                      acAster.quasarBudgetMax
							                          * std::max(0.0f, 1.0f - strataNullMode)
							                          * std::max(0.0f,
							                                     1.0f - 0.5f * uncertainty
							                                         - 0.15f * disagreementNorm)));
							aegisLambdaPredictive =
							    std::max(0.0f,
							             std::min(1.0f,
							                      strataBudget
							                          * (strataPredictiveMode + 0.5f * strataCoupledMode)));
							aegisLambdaOutput =
							    std::max(0.0f,
							             std::min(1.0f,
							                      strataBudget
							                          * (strataOutputMode + 0.5f * strataCoupledMode)));
							aegisLambdaSpatial =
							    std::max(0.0f, std::min(1.0f, 1.0f - aegisLambdaPredictive - aegisLambdaOutput));
							asterMemoryGain *= std::max(0.0f, std::min(1.0f, aegisLambdaOutput));
							strataSelectedExcess = 1.0f - strataNullMode;
						}
						else if (acAster.strataEnabled)
						{
							const float predictiveTrustNorm =
							    rawPredictiveTrust / (rawPredictiveTrust + 1.0f);
							const float outputTrustNorm =
							    rawOutputTrust / (rawOutputTrust + 1.0f);
							const float predictiveErrNorm =
							    std::max(0.0f,
							             std::min(1.0f,
							                      aster.aegisPredictiveErrorEma
							                          / (aegisPredictivePredicted
							                             + aegisPredictiveRealized + 1.0e-6f)));
							const float outputErrNorm =
							    std::max(0.0f,
							             std::min(1.0f,
							                      aster.aegisOutputErrorEma
							                          / (aegisOutputPredicted
							                             + aegisOutputRealized + 1.0e-6f)));
							const float uncertainty =
							    0.5f * (predictiveErrNorm + outputErrNorm);
							const float predictiveGeom =
							    std::max(0.0f,
							             std::min(1.0f,
							                      meritGeometryTrust
							                          * std::max(0.0f, acAster.strataPredictiveGeometryScale)));
							const float coupledGeom =
							    std::max(0.0f,
							             std::min(1.0f,
							                      meritGeometryTrust
							                          * std::max(0.0f, acAster.strataCoupledGeometryScale)));
							const float residualPressure =
							    std::max(predictiveTrustNorm, outputTrustNorm);
							const float nullSignal =
							    std::max(0.0f,
							             acAster.strataNullBias
							                 + 0.55f * uncertainty
							                 + 0.35f * disagreementNorm
							                 - 0.25f * residualPressure);
							const float predictiveSignal =
							    std::max(0.0f,
							             predictiveTrustNorm * std::max(0.0f, 1.0f - predictiveErrNorm)
							                 + 0.25f * predictiveGeom
							                 - 0.10f * uncertainty
							                 - 0.15f * disagreementNorm);
							const float outputSignal =
							    std::max(0.0f,
							             outputTrustNorm * std::max(0.0f, 1.0f - outputErrNorm)
							                 + 0.10f * std::max(0.0f, 1.0f - predictiveGeom)
							                 - 0.10f * uncertainty
							                 - 0.10f * disagreementNorm);
							const float coupledSignal =
							    std::max(0.0f,
							             0.45f * (predictiveTrustNorm + outputTrustNorm)
							                 + 0.30f * std::min(predictiveTrustNorm, outputTrustNorm)
							                 + 0.20f * coupledGeom
							                 - 0.12f * uncertainty
							                 - 0.25f * disagreementNorm);
							const float maxResidualSignal =
							    std::max(predictiveSignal, std::max(outputSignal, coupledSignal));
							const float realizedNullBenefit = nullSignal - maxResidualSignal;
							const float realizedPredictiveBenefit = predictiveSignal - nullSignal;
							const float realizedOutputBenefit = outputSignal - nullSignal;
							const float realizedCoupledBenefit = coupledSignal - nullSignal;
							const float priorNullBenefit = aster.strataNullBenefitEma;
							const float priorPredictiveBenefit = aster.strataPredictiveBenefitEma;
							const float priorOutputBenefit = aster.strataOutputBenefitEma;
							const float priorCoupledBenefit = aster.strataCoupledBenefitEma;
							int previousMode = 0;
							float previousModeWeight = aster.strataLastNullMode;
							if (aster.strataLastPredictiveMode > previousModeWeight)
							{
								previousMode = 1;
								previousModeWeight = aster.strataLastPredictiveMode;
							}
							if (aster.strataLastOutputMode > previousModeWeight)
							{
								previousMode = 2;
								previousModeWeight = aster.strataLastOutputMode;
							}
							if (aster.strataLastCoupledMode > previousModeWeight)
								previousMode = 3;
							float modeScores[4];
							modeScores[0] =
							    priorNullBenefit
							    + 0.25f * realizedNullBenefit
							    + 0.20f * acAster.strataNullBias;
							modeScores[1] =
							    priorPredictiveBenefit
							    + 0.25f * realizedPredictiveBenefit
							    + 0.10f * predictiveGeom;
							modeScores[2] =
							    priorOutputBenefit
							    + 0.25f * realizedOutputBenefit;
							modeScores[3] =
							    priorCoupledBenefit
							    + 0.25f * realizedCoupledBenefit
							    + 0.05f * coupledGeom;
							for (int modeIndex = 0; modeIndex < 4; ++modeIndex)
							{
								if (modeIndex != previousMode)
									modeScores[modeIndex] -= std::max(0.0f, acAster.strataDwellPenalty);
							}
							int selectedMode = 0;
							float selectedScore = modeScores[0];
							for (int modeIndex = 1; modeIndex < 4; ++modeIndex)
							{
								if (modeScores[modeIndex] > selectedScore)
								{
									selectedMode = modeIndex;
									selectedScore = modeScores[modeIndex];
								}
							}
							strataSelectedExcess =
							    (selectedMode == 1)
							        ? realizedPredictiveBenefit
							        : ((selectedMode == 2)
							               ? realizedOutputBenefit
							               : ((selectedMode == 3)
							                      ? realizedCoupledBenefit
							                      : realizedNullBenefit));
							if (selectedMode != 0 && strataSelectedExcess <= 0.0f)
							{
								selectedMode = 0;
								strataSelectedExcess = realizedNullBenefit;
							}
							const float budgetSignal =
							    std::max(0.0f, std::min(1.0f, strataSelectedExcess));
							const float budgetTarget =
							    std::max(0.0f, std::min(1.0f, acAster.strataBudgetMax * budgetSignal));
							strataBudget =
							    (aster.timingBoundaryCount > 0ULL)
							        ? (citadelTrustBeta * aster.strataLastBudget
							           + (1.0f - citadelTrustBeta) * budgetTarget)
							        : budgetTarget;
							strataNullBenefit = realizedNullBenefit;
							strataPredictiveBenefit = realizedPredictiveBenefit;
							strataOutputBenefit = realizedOutputBenefit;
							strataCoupledBenefit = realizedCoupledBenefit;
							strataSwitchRate = (selectedMode == previousMode) ? 0.0f : 1.0f;
							aster.strataNullBenefitEma =
							    (aster.timingBoundaryCount > 0ULL)
							        ? (strataBenefitBeta * aster.strataNullBenefitEma
							           + (1.0f - strataBenefitBeta) * realizedNullBenefit)
							        : realizedNullBenefit;
							aster.strataPredictiveBenefitEma =
							    (aster.timingBoundaryCount > 0ULL)
							        ? (strataBenefitBeta * aster.strataPredictiveBenefitEma
							           + (1.0f - strataBenefitBeta) * realizedPredictiveBenefit)
							        : realizedPredictiveBenefit;
							aster.strataOutputBenefitEma =
							    (aster.timingBoundaryCount > 0ULL)
							        ? (strataBenefitBeta * aster.strataOutputBenefitEma
							           + (1.0f - strataBenefitBeta) * realizedOutputBenefit)
							        : realizedOutputBenefit;
							aster.strataCoupledBenefitEma =
							    (aster.timingBoundaryCount > 0ULL)
							        ? (strataBenefitBeta * aster.strataCoupledBenefitEma
							           + (1.0f - strataBenefitBeta) * realizedCoupledBenefit)
							        : realizedCoupledBenefit;
							if (selectedMode == 1)
							{
								strataNullMode = 0.0f;
								strataPredictiveMode = 1.0f;
								aegisLambdaPredictive = strataBudget;
								aegisLambdaOutput = 0.0f;
							}
							else if (selectedMode == 2)
							{
								strataNullMode = 0.0f;
								strataOutputMode = 1.0f;
								aegisLambdaPredictive = 0.0f;
								aegisLambdaOutput = strataBudget;
							}
							else if (selectedMode == 3)
							{
								strataNullMode = 0.0f;
								strataCoupledMode = 1.0f;
								const float predictiveShare =
								    predictiveTrustNorm / (predictiveTrustNorm + outputTrustNorm + 1.0e-6f);
								aegisLambdaPredictive =
								    std::max(0.0f, std::min(1.0f, strataBudget * predictiveShare));
								aegisLambdaOutput =
								    std::max(0.0f, std::min(1.0f, strataBudget - aegisLambdaPredictive));
							}
							else
							{
								aegisLambdaPredictive = 0.0f;
								aegisLambdaOutput = 0.0f;
							}
							aegisLambdaSpatial =
							    std::max(0.0f, std::min(1.0f, 1.0f - aegisLambdaPredictive - aegisLambdaOutput));
							meritSparrowTrust = aegisLambdaPredictive;
							asterMemoryGain *= std::max(0.0f, std::min(1.0f, aegisLambdaOutput));
						}
						else if (acAster.meritEnabled)
						{
							const float predictiveTrustNorm =
							    rawPredictiveTrust / (rawPredictiveTrust + 1.0f);
							const float outputTrustNorm =
							    rawOutputTrust / (rawOutputTrust + 1.0f);
							const float predictiveErrNorm =
							    std::max(0.0f,
							             std::min(1.0f,
							                      aster.aegisPredictiveErrorEma
							                          / (aegisPredictivePredicted
							                             + aegisPredictiveRealized + 1.0e-6f)));
							const float outputErrNorm =
							    std::max(0.0f,
							             std::min(1.0f,
							                      aster.aegisOutputErrorEma
							                          / (aegisOutputPredicted
							                             + aegisOutputRealized + 1.0e-6f)));
							const float uncertainty =
							    0.5f * (predictiveErrNorm + outputErrNorm);
							const float contextPenalty = 0.0f;
							const float tauSignal =
							    std::max(0.0f,
							             std::min(1.0f,
							                      0.45f * uncertainty
							                          + 0.30f * disagreementNorm
							                          - 0.30f * meritGeometryTrust
							                          + 0.25f * contextPenalty));
							const float tauTarget =
							    acAster.meritTauMin
							    + (acAster.meritTauMax - acAster.meritTauMin) * tauSignal;
							meritTau =
							    (aster.timingBoundaryCount > 0ULL)
							        ? (citadelTrustBeta * aster.meritLastTau
							           + (1.0f - citadelTrustBeta) * tauTarget)
							        : tauTarget;
							meritCovariance =
							    std::max(0.0f,
							             std::min(0.95f,
							                      acAster.meritCovarianceMix
							                          * std::min(predictiveTrustNorm, outputTrustNorm)
							                          * std::max(0.0f, 1.0f - disagreementNorm)));
							const float coupling =
							    meritCovariance
							    * sqrtf(std::max(0.0f, rawPredictiveTrust * rawOutputTrust));
							const float h11 = std::max(1.0e-6f, meritTau + rawPredictiveTrust);
							const float h22 = std::max(1.0e-6f, meritTau + rawOutputTrust);
							const float det = std::max(1.0e-6f, h11 * h22 - coupling * coupling);
							float wPredictive =
							    (rawPredictiveTrust * predictiveTrustNorm * h22
							     - coupling * rawOutputTrust * outputTrustNorm)
							    / det;
							float wOutput =
							    (h11 * rawOutputTrust * outputTrustNorm
							     - coupling * rawPredictiveTrust * predictiveTrustNorm)
							    / det;
							wPredictive = std::max(0.0f, wPredictive);
							wOutput = std::max(0.0f, wOutput);
							const float weightSum = wPredictive + wOutput + 1.0e-6f;
							wPredictive /= weightSum;
							wOutput /= weightSum;
							const float budgetSignal =
							    std::max(0.0f,
							             std::min(1.0f,
							                      0.45f * (0.5f * (predictiveTrustNorm + outputTrustNorm))
							                          + 0.35f * meritGeometryTrust
							                          - 0.15f * disagreementNorm
							                          - 0.15f * uncertainty
							                          - 0.20f * contextPenalty));
							const float budgetTarget =
							    std::max(0.0f, std::min(1.0f, acAster.meritBudgetMax * budgetSignal));
							meritBudget =
							    (aster.timingBoundaryCount > 0ULL)
							        ? (citadelTrustBeta * aster.meritLastBudget
							           + (1.0f - citadelTrustBeta) * budgetTarget)
							        : budgetTarget;
							const float residualNorm =
							    sqrtf(std::max(0.0f,
							                   wPredictive * wPredictive + wOutput * wOutput
							                       + 2.0f * meritCovariance * wPredictive * wOutput));
							const float budgetScale =
							    (residualNorm > 1.0e-6f)
							        ? std::max(0.0f, std::min(1.0f, meritBudget / residualNorm))
							        : 0.0f;
							aegisLambdaPredictive = std::max(0.0f, std::min(1.0f, wPredictive * budgetScale));
							aegisLambdaOutput = std::max(0.0f, std::min(1.0f, wOutput * budgetScale));
							aegisLambdaSpatial =
							    std::max(0.0f, std::min(1.0f, 1.0f - aegisLambdaPredictive - aegisLambdaOutput));
							meritSparrowTrust = aegisLambdaPredictive;
							asterMemoryGain *= std::max(0.0f, std::min(1.0f, aegisLambdaOutput));
						}
						else if (acAster.rampartEnabled)
						{
							const float predictiveTrustNorm =
							    rawPredictiveTrust / (rawPredictiveTrust + 1.0f);
							const float outputTrustNorm =
							    rawOutputTrust / (rawOutputTrust + 1.0f);
							const float predictiveErrNorm =
							    std::max(0.0f,
							             std::min(1.0f,
							                      aster.aegisPredictiveErrorEma
							                          / (aegisPredictivePredicted
							                             + aegisPredictiveRealized + 1.0e-6f)));
							const float outputErrNorm =
							    std::max(0.0f,
							             std::min(1.0f,
							                      aster.aegisOutputErrorEma
							                          / (aegisOutputPredicted
							                             + aegisOutputRealized + 1.0e-6f)));
							const float uncertainty =
							    0.5f * (predictiveErrNorm + outputErrNorm);
							const float tauSignal =
							    std::max(0.0f, std::min(1.0f, 0.5f * (uncertainty + disagreementNorm)));
							const float tauTarget =
							    acAster.rampartTauMin
							    + (acAster.rampartTauMax - acAster.rampartTauMin) * tauSignal;
							rampartTau =
							    (aster.timingBoundaryCount > 0ULL)
							        ? (citadelTrustBeta * aster.rampartLastTau
							           + (1.0f - citadelTrustBeta) * tauTarget)
							        : tauTarget;
							rampartCovariance =
							    std::max(0.0f,
							             std::min(0.95f,
							                      acAster.rampartCovarianceMix
							                          * std::min(predictiveTrustNorm, outputTrustNorm)
							                          * std::max(0.0f, 1.0f - disagreementNorm)));
							const float coupling =
							    rampartCovariance
							    * sqrtf(std::max(0.0f, rawPredictiveTrust * rawOutputTrust));
							const float h11 = std::max(1.0e-6f, rampartTau + rawPredictiveTrust);
							const float h22 = std::max(1.0e-6f, rampartTau + rawOutputTrust);
							const float det = std::max(1.0e-6f, h11 * h22 - coupling * coupling);
							float wSpatial = (rampartTau + spatialPrecision)
							               / std::max(1.0e-6f, rampartTau + spatialPrecision);
							float wPredictive =
							    (rawPredictiveTrust * predictiveTrustNorm * h22
							     - coupling * rawOutputTrust * outputTrustNorm)
							    / det;
							float wOutput =
							    (h11 * rawOutputTrust * outputTrustNorm
							     - coupling * rawPredictiveTrust * predictiveTrustNorm)
							    / det;
							wSpatial = std::max(0.0f, wSpatial);
							wPredictive = std::max(0.0f, wPredictive);
							wOutput = std::max(0.0f, wOutput);
							const float weightSum = wSpatial + wPredictive + wOutput + 1.0e-6f;
							wSpatial /= weightSum;
							wPredictive /= weightSum;
							wOutput /= weightSum;
							const float contextPenalty = 0.0f;
							const float budgetSignal =
							    std::max(0.0f,
							             std::min(1.0f,
							                      0.5f * (predictiveTrustNorm + outputTrustNorm)
							                          - 0.25f * disagreementNorm
							                          - 0.25f * uncertainty
							                          - 0.25f * contextPenalty));
							const float budgetTarget =
							    std::max(0.0f, std::min(1.0f, acAster.rampartBudgetMax * budgetSignal));
							rampartBudget =
							    (aster.timingBoundaryCount > 0ULL)
							        ? (citadelTrustBeta * aster.rampartLastBudget
							           + (1.0f - citadelTrustBeta) * budgetTarget)
							        : budgetTarget;
							const float residualNorm =
							    sqrtf(std::max(0.0f,
							                   wPredictive * wPredictive + wOutput * wOutput
							                       + 2.0f * rampartCovariance * wPredictive * wOutput));
							const float budgetScale =
							    (residualNorm > 1.0e-6f)
							        ? std::max(0.0f, std::min(1.0f, rampartBudget / residualNorm))
							        : 0.0f;
							aegisLambdaPredictive = std::max(0.0f, std::min(1.0f, wPredictive * budgetScale));
							aegisLambdaOutput = std::max(0.0f, std::min(1.0f, wOutput * budgetScale));
							aegisLambdaSpatial =
							    std::max(0.0f, std::min(1.0f, 1.0f - aegisLambdaPredictive - aegisLambdaOutput));
							rampartSparrowTrust = aegisLambdaPredictive;
							asterMemoryGain *= std::max(0.0f, std::min(1.0f, aegisLambdaOutput));
						}
						else if (acAster.citadelEnabled)
						{
							const float predictiveTrustNorm =
							    aster.citadelPredictiveTrustEma / (aster.citadelPredictiveTrustEma + 1.0f);
							const float outputTrustNorm =
							    aster.citadelOutputTrustEma / (aster.citadelOutputTrustEma + 1.0f);
							const float predictiveDominance =
							    std::max(0.0f, predictiveTrustNorm - outputTrustNorm);
							const float citadelConflict =
							    disagreementNorm * predictiveDominance;
							const float anchorTarget =
							    std::max(0.0f,
							             std::min(0.95f,
							                      acAster.citadelAnchorBase
							                          + acAster.citadelDisagreementScale * citadelConflict));
							citadelAnchor =
							    (aster.timingBoundaryCount > 0ULL)
							        ? (citadelTrustBeta * aster.citadelLastAnchor
							           + (1.0f - citadelTrustBeta) * anchorTarget)
							        : anchorTarget;
							residualScale = std::max(0.0f, 1.0f - citadelAnchor);
							spatialPrecision +=
							    std::max(0.0f, acAster.citadelSpatialScale) * citadelAnchor;
						}
						const float predictivePrecision =
						    residualScale
						    * rawPredictiveTrust;
						const float outputPrecision =
						    residualScale
						    * rawOutputTrust;
						if (!acAster.rampartEnabled && !acAster.meritEnabled && !acAster.strataEnabled
						    && !acAster.auroraEnabled && !acAster.seamEnabled && !acAster.quasarEnabled)
						{
							const float totalPrecision =
							    spatialPrecision + predictivePrecision + outputPrecision + 1e-6f;
							aegisLambdaSpatial = spatialPrecision / totalPrecision;
							aegisLambdaPredictive = predictivePrecision / totalPrecision;
							aegisLambdaOutput = outputPrecision / totalPrecision;
							if (acAster.citadelEnabled)
							{
								const float predictiveTrustNorm =
								    aster.citadelPredictiveTrustEma / (aster.citadelPredictiveTrustEma + 1.0f);
								citadelSparrowTrust =
								    std::max(0.0f,
								             std::min(1.0f,
								                      predictiveTrustNorm
								                          * std::max(0.0f, 1.0f - citadelAnchor)));
							}
							asterMemoryGain *= std::max(0.0f, std::min(1.0f, aegisLambdaOutput));
						}
					}

					if (asterMemoryGain > 0.0f)
					{
						const float batchScale = static_cast<float>(tensorDff.batchCount) * asterMemoryGain;
						for (unsigned int j = 0; j < outputDim; ++j)
						{
							const float predictedResidual =
							    clipf_maybe(predictedResidualNextW[j] * residualStd[j], gradClip);
							if (!is_finite(predictedResidual))
								continue;
							if (j < outTr.gBias.size())
								outTr.gBias[j] += batchScale * predictedResidual;
							const size_t rowOff = static_cast<size_t>(j) * static_cast<size_t>(outTr.in);
							for (unsigned int i = 0; i < outputHiddenDim; ++i)
								outTr.gW[rowOff + i] += batchScale * predictedResidual * rawHiddenMean[outputHiddenOffset + i];
						}
					}

					for (unsigned int m = 0; m < stateRank; ++m)
					{
						if (m >= aster.poleNumer.size() || m >= aster.poleDenom.size()
						    || m >= aster.pole.size() || m >= aster.latent.size())
							continue;
						const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(stateFeatureDim);
						const float diagPole = (m < stateRank) ? stateModel[rowOff + m] : 0.0f;
						aster.poleNumer[m] = diagPole;
						aster.poleDenom[m] = 1.0f;
						aster.pole[m] = std::max(-acAster.asterPoleMax,
						                         std::min(acAster.asterPoleMax, diagPole));
						aster.latent[m] = filteredState[m];
					}

					unsigned int asterActiveModes = 0u;
					double nextEffectiveSigmaSq = 0.0;
					for (unsigned int m = 0; m < stateRank; ++m)
					{
						const float sigma = (m < nextSigma.size()) ? nextSigma[m] : 0.0f;
						if (m < aster.sigma.size())
							aster.sigma[m] = sigma;
						nextEffectiveSigmaSq += static_cast<double>(sigma) * static_cast<double>(sigma);
						if (sigma > acAster.asterEdgeThreshold)
							asterActiveModes += 1u;
					}
					aster.leftMode.swap(nextLeft);
					aster.rightMode.swap(nextRight);
					aster.lastActiveModes = asterActiveModes;
					aster.lastEdge = (stateRank > 0u) ? std::max(0.0f, aster.sigma[0]) : 0.0f;
					aster.lastSecondEdge = (stateRank > 1u) ? std::max(0.0f, aster.sigma[1]) : 0.0f;
					aster.lastSecondEdgeRatio = (aster.lastEdge > sigmaEps)
					                          ? std::max(0.0f, aster.lastSecondEdge / (aster.lastEdge + sigmaEps))
					                          : 0.0f;
					aster.lastSigma = static_cast<float>(sqrt(nextEffectiveSigmaSq));
					aster.lastPredR2 = asterPredR2;
					aster.lastMemoryGain = asterMemoryGain;
					aster.aegisPrevPredictiveScore = aegisPredictiveRealized;
					aster.aegisPrevOutputScore = aegisOutputRealized;
					aster.aegisLastLambdaSpatial = aegisLambdaSpatial;
					aster.aegisLastLambdaPredictive = aegisLambdaPredictive;
					aster.aegisLastLambdaOutput = aegisLambdaOutput;
					aster.aegisLastPredictivePredicted = aegisPredictivePredicted;
					aster.aegisLastPredictiveRealized = aegisPredictiveRealized;
					aster.aegisLastOutputPredicted = aegisOutputPredicted;
					aster.aegisLastOutputRealized = aegisOutputRealized;
					aster.aegisLastChannelDisagreement =
					    fabsf(aegisPredictiveRealized - aegisOutputRealized);
					aster.citadelLastAnchor = citadelAnchor;
					aster.citadelLastHardRegimeMass = citadelHardRegimeMass;
					aster.citadelLastSparrowTrust = citadelSparrowTrust;
					aster.rampartLastTau = rampartTau;
					aster.rampartLastBudget = rampartBudget;
					aster.rampartLastCovariance = rampartCovariance;
					aster.rampartLastSparrowTrust = rampartSparrowTrust;
					aster.meritLastTau = meritTau;
					aster.meritLastBudget = meritBudget;
					aster.meritLastCovariance = meritCovariance;
					aster.meritLastSparrowTrust = meritSparrowTrust;
					aster.meritLastGeometryTrust = meritGeometryTrust;
					aster.strataLastNullMode = strataNullMode;
					aster.strataLastPredictiveMode = strataPredictiveMode;
					aster.strataLastOutputMode = strataOutputMode;
					aster.strataLastCoupledMode = strataCoupledMode;
					aster.strataLastBudget = strataBudget;
					aster.strataLastNullBenefit = strataNullBenefit;
					aster.strataLastPredictiveBenefit = strataPredictiveBenefit;
					aster.strataLastOutputBenefit = strataOutputBenefit;
					aster.strataLastCoupledBenefit = strataCoupledBenefit;
					aster.strataLastSelectedExcess = strataSelectedExcess;
					aster.strataLastSwitchRate = strataSwitchRate;
					for (unsigned int i = 0; i < controlDim; ++i)
						aster.prevControlMean[i] = controlMean[i];
					for (unsigned int j = 0; j < outputDim; ++j)
						aster.prevResidualMean[j] = residualMean[j];
					const timespec asterBoundaryEnd = monotonic_now();
					aster.timingBoundaryCount += 1ULL;
					aster.totalBoundaryNs += monotonic_elapsed_ns(asterBoundaryStart, asterBoundaryEnd);
					aster.totalSetupNs += asterSetupNs;
					aster.totalTransportNs += asterTransportNs;
					aster.totalTransferFitNs += asterTransferFitNs;
					aster.totalStateFitNs += asterStateFitNs;
					aster.totalInnovationFitNs += asterInnovationFitNs;
					aster.totalApplyNs += monotonic_elapsed_ns(asterApplyStart, asterBoundaryEnd);
				}

					// Optional global grad-norm clipping (modern feature).
					// We compute the L2 norm of the (averaged) gradients including L1/L2 decay terms
					// (for parity with the actual update direction), then scale all gradients by:
					//   scale = min(1, clipNorm / (norm + eps))
					//
					// Defaults preserve behavior (globalGradClipNorm == 0 disables).
					float gradNorm = 0.0f;
					float gradScale = 1.0f;
					const float clipNorm = trainingConfig.globalGradClipNorm;
					if (clipNorm > 0.0f)
					{
						double sumsq = 0.0;
						for (unsigned int t = 0; t < tensorDff.T.size(); ++t)
						{
							const TensorDFFState::Transition& tr = tensorDff.T[t];
							const float wd1 = skeleton->getWeightDecay1(t);
							const float wd2 = skeleton->getWeightDecay2(t);

							for (size_t idx = 0; idx < tr.W.size(); ++idx)
							{
								float g = tr.gW[idx] * invBatch;
								if ((wd1 != 0.0f) || (wd2 != 0.0f))
								{
									const float w = tr.W[idx];
									const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
									g += (wd1 * wSign) + (wd2 * w);
								}
								const double gd = static_cast<double>(g);
								sumsq += gd * gd;
							}

							// Bias grads (no decay, but included in the global norm).
							for (unsigned int j = 0; j < tr.gBias.size(); ++j)
							{
								const float gB = tr.gBias[j] * invBatch;
								const double bd = static_cast<double>(gB);
								sumsq += bd * bd;
							}
						}

						if (!is_finite_double(sumsq))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite grad-norm accumulation detected (NaN/Inf)");
							running = false;
							return;
						}
						gradNorm = static_cast<float>(sqrt(sumsq));
						const float eps = 1e-12f;
						if (gradNorm > clipNorm)
							gradScale = clipNorm / (gradNorm + eps);
						else
							gradScale = 1.0f;
					}
					lastGradNorm = gradNorm;
					lastGradNormScale = gradScale;
					if (!is_finite(lastGradNorm) || !is_finite(lastGradNormScale) || lastGradNormScale <= 0.0f)
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite grad clipping metadata detected (NaN/Inf)");
						running = false;
						return;
					}

				if (trainingConfig.optimizer.type == OptimizerConfig::ATLAS)
				{
					// ATLAS update
					const ATLASConfig& ac = trainingConfig.atlas;

					// Ensure ATLAS state is initialized
					if (tensorDff.atlasState.size() != tensorDff.T.size())
						tensorDff.atlasState.resize(tensorDff.T.size());
					const float dffSparrowTrust =
					    ((ac.citadelEnabled || ac.rampartEnabled || ac.meritEnabled || ac.strataEnabled
					      || ac.auroraEnabled || ac.seamEnabled || ac.quasarEnabled) && tensorDff.aster.initialized)
					        ? std::max(0.0f,
					                   std::min(1.0f,
					                            (ac.strataEnabled || ac.auroraEnabled || ac.seamEnabled || ac.quasarEnabled)
					                                ? tensorDff.aster.aegisLastLambdaPredictive
					                                : (ac.meritEnabled
					                                ? tensorDff.aster.meritLastSparrowTrust
					                                : (ac.rampartEnabled
					                                       ? tensorDff.aster.rampartLastSparrowTrust
					                                       : tensorDff.aster.citadelLastSparrowTrust))))
					        : 1.0f;

					for (unsigned int t = 0; t < tensorDff.T.size(); ++t)
					{
						TensorDFFState::Transition& tr = tensorDff.T[t];
						tensorDff.atlasState[t].externalSparrowTrust = dffSparrowTrust;
						const float lr = skeleton->getLearningRate(t) * lrScheduleMultiplier;
						const float wd1 = skeleton->getWeightDecay1(t);
						const float wd2 = skeleton->getWeightDecay2(t);

						if (!atlas::update(tensorDff.atlasState[t],
							&tr.W[0], &tr.gW[0],
							tr.out, tr.in,
							invBatch, lr, wd1, wd2, gradScale,
							ac, rngEngine, getLogger(), "dff.W"))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							    "SGDHelper_DFF: ATLAS weight update entered NaN recovery");
							running = false;
							return;
						}

						if (!atlas::updateBias(&tr.bias[0], &tr.gBias[0],
						                       tr.out, invBatch, lr, gradScale))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR,
							    "SGDHelper_DFF: ATLAS bias update produced NaN/Inf");
							running = false;
							return;
						}
					}

					tensorDff.batchCount = 0;
				}
				else
				{
				for (unsigned int t = 0; t < tensorDff.T.size(); ++t)
				{
					TensorDFFState::Transition& tr = tensorDff.T[t];
						const float lr = skeleton->getLearningRate(t) * lrScheduleMultiplier;
					const float mf = skeleton->getMomentumFactor(t);
					const float wd1 = skeleton->getWeightDecay1(t);
					const float wd2 = skeleton->getWeightDecay2(t);

					for (size_t idx = 0; idx < tr.W.size(); ++idx)
					{
							float g = tr.gW[idx] * invBatch;
						// L1/L2 decay on weight (matches Node::getDelta semantics after bugfix).
						if ((wd1 != 0.0f) || (wd2 != 0.0f))
						{
							const float w = tr.W[idx];
							const float wSign = (w > 0.0f) ? 1.0f : ((w < 0.0f) ? -1.0f : 0.0f);
							g += (wd1 * wSign) + (wd2 * w);
						}
							// Apply optional global grad clipping scale.
							g *= gradScale;
						if (!is_finite(g))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite weight gradient detected (NaN/Inf)");
							running = false;
							return;
						}

						const float v = (mf * tr.vW[idx]) + (lr * g);
						tr.vW[idx] = v;
						tr.W[idx] -= v;
						tr.gW[idx] = 0.0f;
						if (!is_finite(tr.W[idx]) || !is_finite(tr.vW[idx]))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite weight update detected (NaN/Inf)");
							running = false;
							return;
						}
					}

					// Per-neuron bias (no momentum/weight decay).
					{
						for (unsigned int j = 0; j < tr.out; ++j)
						{
								float gB = (j < tr.gBias.size()) ? (tr.gBias[j] * invBatch) : 0.0f;
								gB *= gradScale;
							if (!is_finite(gB))
							{
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite bias gradient detected (NaN/Inf)");
								running = false;
								return;
							}
							if (j < tr.bias.size())
								tr.bias[j] -= (lr * gB);
							if (j < tr.gBias.size())
								tr.gBias[j] = 0.0f;
							if (j < tr.bias.size() && !is_finite(tr.bias[j]))
							{
								lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_DFF: non-finite bias update detected (NaN/Inf)");
								running = false;
								return;
							}
						}
					}
				}

				tensorDff.batchCount = 0;
				}
			}

			// Save the autotuning data (kept for parity with old path)
			{
				const float learningRate = skeleton->getLearningRate(lastTransition) * lrScheduleMultiplier;
				shmea::GList nbRow;
				nbRow.addFloat(overallTotalAccuracy);
				nbRow.addFloat(learningRate);
				nbRecord.addRow(nbRow);
			}
		}
	}
}
