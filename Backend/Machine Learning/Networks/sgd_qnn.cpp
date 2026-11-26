// Net-type-specific SGD: Quaternion Neural Network (TYPE_QNN) dense tensor path.
#include "network.h"
#include "sgd_utils.h"

#include "Backend/Database/GLogger.h"

#include "../DataObjects/DataInput.h"
#include "../GMath/gmath.h"
#include "../GMath/quaternion.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <vector>

#include "logfmt_utils.h"

using namespace glades;
using namespace glades::logfmt;

namespace {

inline float signf_nonzero(float w)
{
	if (w > 0.0f) return 1.0f;
	if (w < 0.0f) return -1.0f;
	return 0.0f;
}

inline void accumulate_q_grad_norm_sq(const Quaternion& w,
                                      const Quaternion& gAccum,
                                      float invBatch,
                                      float wd1,
                                      float wd2,
                                      bool includeDecay,
                                      double& sumsq)
{
	const float ws[4] = { w.r, w.i, w.j, w.k };
	const float gs[4] = { gAccum.r, gAccum.i, gAccum.j, gAccum.k };
	for (unsigned int c = 0u; c < 4u; ++c)
	{
		float g = gs[c] * invBatch;
		if (includeDecay && ((wd1 != 0.0f) || (wd2 != 0.0f)))
			g += (wd1 * signf_nonzero(ws[c])) + (wd2 * ws[c]);
		const double gd = static_cast<double>(g);
		sumsq += gd * gd;
	}
}

inline bool adamw_update_quaternion(Quaternion& w,
                                    Quaternion& m,
                                    Quaternion& v,
                                    const Quaternion& gAccum,
                                    float invBatch,
                                    float gradScale,
                                    float lr,
                                    float wd1,
                                    float wd2,
                                    bool applyDecay,
                                    float beta1,
                                    float beta2,
                                    float bc1,
                                    float bc2,
                                    float epsAdam)
{
	float* wp[4] = { &w.r, &w.i, &w.j, &w.k };
	float* mp[4] = { &m.r, &m.i, &m.j, &m.k };
	float* vp[4] = { &v.r, &v.i, &v.j, &v.k };
	const float gs[4] = { gAccum.r, gAccum.i, gAccum.j, gAccum.k };
	for (unsigned int c = 0u; c < 4u; ++c)
	{
		float g = gs[c] * invBatch;
		if (applyDecay && wd1 != 0.0f)
			g += wd1 * signf_nonzero(*wp[c]);
		g *= gradScale;
		if (!glades::sgd_detail::is_finite(g))
			return false;

		*mp[c] = (beta1 * (*mp[c])) + ((1.0f - beta1) * g);
		*vp[c] = (beta2 * (*vp[c])) + ((1.0f - beta2) * g * g);
		const float mHat = (*mp[c]) / bc1;
		const float vHat = (*vp[c]) / bc2;
		const float decayTerm = applyDecay ? (wd2 * (*wp[c])) : 0.0f;
		*wp[c] -= lr * ((mHat / (sqrtf(vHat) + epsAdam)) + decayTerm);

		if (!glades::sgd_detail::is_finite(*wp[c]) ||
		    !glades::sgd_detail::is_finite(*mp[c]) ||
		    !glades::sgd_detail::is_finite(*vp[c]))
			return false;
	}
	return true;
}

inline bool sgd_update_quaternion(Quaternion& w,
                                  Quaternion& vel,
                                  const Quaternion& gAccum,
                                  float invBatch,
                                  float gradScale,
                                  float lr,
                                  float mf,
                                  float wd1,
                                  float wd2,
                                  bool applyDecay)
{
	float* wp[4] = { &w.r, &w.i, &w.j, &w.k };
	float* vp[4] = { &vel.r, &vel.i, &vel.j, &vel.k };
	const float gs[4] = { gAccum.r, gAccum.i, gAccum.j, gAccum.k };
	for (unsigned int c = 0u; c < 4u; ++c)
	{
		float g = gs[c] * invBatch;
		if (applyDecay && ((wd1 != 0.0f) || (wd2 != 0.0f)))
			g += (wd1 * signf_nonzero(*wp[c])) + (wd2 * (*wp[c]));
		g *= gradScale;
		if (!glades::sgd_detail::is_finite(g))
			return false;

		const float nextV = (mf * (*vp[c])) + (lr * g);
		*vp[c] = nextV;
		*wp[c] -= nextV;

		if (!glades::sgd_detail::is_finite(*wp[c]) || !glades::sgd_detail::is_finite(*vp[c]))
			return false;
	}
	return true;
}

} // namespace

void glades::NNetwork::SGDHelper_QNN(unsigned int inputRowCounter, int runType)
{
	using namespace glades::sgd_detail;

	const bool isTrain = (runType == RUN_TRAIN);
	const unsigned int dataSize = isTrain ? (di ? di->getTrainSize() : 0u) : (di ? di->getTestSize() : 0u);
	const float gradClip = trainingConfig.perElementGradClip;

	if (!ensureTensorParametersInitialized())
	{
		running = false;
		return;
	}
	if (!tensorQnn.initialized || tensorQnn.qSizes.size() < 2u || tensorQnn.T.empty())
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_QNN: tensor state is not initialized");
		running = false;
		return;
	}

	const unsigned int realIn = tensorQnn.realInputSize;
	const unsigned int realOut = tensorQnn.realOutSize;
	const unsigned int qIn = tensorQnn.qSizes[0];
	const int H = skeleton->numHiddenLayers();
	const unsigned int numLayers = static_cast<unsigned int>(tensorQnn.qSizes.size());
	const unsigned int lastLayer = numLayers - 1u;
	const unsigned int lastTransition = static_cast<unsigned int>(tensorQnn.T.size() - 1u);

	const bool packGroupsOf4 =
	    (tensorQnn.inputPackingMode == static_cast<unsigned int>(QuaternionConfig::PACK_GROUPS_OF_4));
	const bool splitReadout =
	    (tensorQnn.outputReadoutMode == static_cast<unsigned int>(QuaternionConfig::READOUT_SPLIT_COMPONENTS));

	// Inverted dropout masks per quaternion node (layer 0 = input, layers 1..H = hidden).
	std::vector< std::vector<unsigned char> > keepMasks;
	keepMasks.resize(static_cast<size_t>(H) + 1u);
	std::vector<float> keepScales;
	keepScales.assign(keepMasks.size(), 1.0f);
	{
		const float pIn = skeleton->getPInput();
		keepMasks[0].assign(qIn, 1u);
		if (isTrain && pIn > 0.0f)
		{
			if (pIn < 1.0f)
				keepScales[0] = 1.0f / (1.0f - pIn);
			for (unsigned int q = 0u; q < qIn; ++q)
				keepMasks[0][q] = (glades::rng::uniform_double(rngEngine, 0.0, 1.0) >= static_cast<double>(pIn)) ? 1u : 0u;
		}
	}
	for (int h = 0; h < H; ++h)
	{
		const unsigned int layerIdx = static_cast<unsigned int>(h) + 1u;
		const unsigned int sz = (layerIdx < tensorQnn.qSizes.size()) ? tensorQnn.qSizes[layerIdx] : 0u;
		keepMasks[layerIdx].assign(sz, 1u);
		const float p = skeleton->getPDropout(static_cast<unsigned int>(h));
		if (isTrain && p > 0.0f)
		{
			if (p < 1.0f && layerIdx < keepScales.size())
				keepScales[layerIdx] = 1.0f / (1.0f - p);
			for (unsigned int q = 0u; q < sz; ++q)
				keepMasks[layerIdx][q] = (glades::rng::uniform_double(rngEngine, 0.0, 1.0) >= static_cast<double>(p)) ? 1u : 0u;
		}
	}

	results.clear();
	if (isTrain)
		nbRecord.clear();

	// === Forward Pass ===
	const int costFx = skeleton->getOutputType();
	const bool useSoftmax = ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL)) && (realOut > 1u);
	const bool recordActivations = (di && (dataSize > 0u) && (inputRowCounter == dataSize - 1u));

	{
		const float* xData = NULL;
		unsigned int xSize = 0u;
		if (di)
		{
			if (isTrain)
				di->getTrainRowView(inputRowCounter, xData, xSize);
			else
				di->getTestRowView(inputRowCounter, xData, xSize);
		}
		if (!span_all_finite_bounded(xData, xSize, 32u))
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_QNN: non-finite input data detected (NaN/Inf)");
			running = false;
			return;
		}

		// Encode real features -> input quaternions a[0].
		const QuaternionInputPackingMode packingMode =
		    packGroupsOf4 ? QUAT_PACK_GROUPS_OF_4 : QUAT_ONE_PER_FEATURE;
		if (qIn > 0u)
			encodeRealToQuaternions(xData, xSize, realIn, packingMode, &tensorQnn.a[0][0], qIn);

		// Optional input dropout (per quaternion node).
		if (isTrain && keepScales[0] != 1.0f)
		{
			for (unsigned int q = 0u; q < qIn; ++q)
			{
				if (keepMasks[0][q] == 0u)
					tensorQnn.a[0][q] = Quaternion::zero();
				else
					tensorQnn.a[0][q] *= keepScales[0];
			}
		}
		else if (isTrain)
		{
			for (unsigned int q = 0u; q < qIn; ++q)
			{
				if (keepMasks[0][q] == 0u)
					tensorQnn.a[0][q] = Quaternion::zero();
			}
		}

		// Dense Hamilton-product transitions.
		for (unsigned int t = 0u; t < tensorQnn.T.size(); ++t)
		{
			const TensorQNNState::Transition& tr = tensorQnn.T[t];
			const int actFx = skeleton->getActivationType(t);
			const float actParam = skeleton->getActivationParam(t);
			const bool softmaxThisLayer = useSoftmax && (t == lastTransition);
			const bool dropoutThisLayer = isTrain && (t < static_cast<unsigned int>(H));
			const unsigned int outLayerIdx = t + 1u;
			const float dropoutScale =
			    (dropoutThisLayer && outLayerIdx < keepScales.size()) ? keepScales[outLayerIdx] : 1.0f;

			for (unsigned int m = 0u; m < tr.qOut; ++m)
			{
				if (dropoutThisLayer &&
				    outLayerIdx < keepMasks.size() &&
				    m < keepMasks[outLayerIdx].size() &&
				    keepMasks[outLayerIdx][m] == 0u)
				{
					tensorQnn.z[outLayerIdx][m] = Quaternion::zero();
					tensorQnn.a[outLayerIdx][m] = Quaternion::zero();
					continue;
				}

				Quaternion zq = (m < tr.bias.size()) ? tr.bias[m] : Quaternion::zero();
				const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(tr.qIn);
				for (unsigned int n = 0u; n < tr.qIn; ++n)
					zq += Quaternion::mul(tr.W[rowOff + n], tensorQnn.a[t][n]);

				if (!zq.isFinite())
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_QNN: non-finite pre-activation detected (NaN/Inf)");
					running = false;
					return;
				}

				tensorQnn.z[outLayerIdx][m] = zq;
				if (recordActivations)
					cNodeActivations.addFloat(zq.r);

				if (softmaxThisLayer)
				{
					tensorQnn.a[outLayerIdx][m] = zq;
				}
				else
				{
					Quaternion aq(
					    GMath::squash(zq.r, actFx, actParam),
					    GMath::squash(zq.i, actFx, actParam),
					    GMath::squash(zq.j, actFx, actParam),
					    GMath::squash(zq.k, actFx, actParam));
					if (!aq.isFinite())
					{
						lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_QNN: non-finite activation detected (NaN/Inf)");
						running = false;
						return;
					}
					if (dropoutThisLayer && dropoutScale != 1.0f)
						aq *= dropoutScale;
					tensorQnn.a[outLayerIdx][m] = aq;
				}
			}
		}

		// Readout quaternion outputs -> real scalar vector outReadout[0..realOut-1].
		if (tensorQnn.outReadout.size() != realOut)
			tensorQnn.outReadout.assign(realOut, 0.0f);

		const QuaternionOutputReadoutMode readoutMode =
		    splitReadout ? QUAT_READOUT_SPLIT_COMPONENTS : QUAT_READOUT_REAL_PART;
		const unsigned int qOut = static_cast<unsigned int>(tensorQnn.z[lastLayer].size());

		if (useSoftmax)
		{
			std::vector<float> outLogits(realOut, 0.0f);
			if (qOut > 0u && realOut > 0u)
				decodeQuaternionsToReal(&tensorQnn.z[lastLayer][0], qOut, readoutMode, &outLogits[0], realOut);
			std::vector<float> outProbs;
			softmax_stable(outLogits, outProbs);
			for (unsigned int k = 0u; k < realOut; ++k)
			{
				if (!is_finite(outProbs[k]))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_QNN: non-finite softmax probability detected (NaN/Inf)");
					running = false;
					return;
				}
				tensorQnn.outReadout[k] = outProbs[k];
			}
		}
		else if (qOut > 0u && realOut > 0u)
		{
			decodeQuaternionsToReal(&tensorQnn.a[lastLayer][0], qOut, readoutMode, &tensorQnn.outReadout[0], realOut);
		}

		if (recordActivations)
			cNodeActivations.addString(",");
	}

	// === Output Bookkeeping & Metrics ===
	const float* yData = NULL;
	unsigned int ySize = 0u;
	if (di)
	{
		if (isTrain)
			di->getTrainExpectedRowView(inputRowCounter, yData, ySize);
		else
			di->getTestExpectedRowView(inputRowCounter, yData, ySize);
	}
	if (!span_all_finite_bounded(yData, ySize, 32u))
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_QNN: non-finite expected outputs detected (NaN/Inf)");
		running = false;
		return;
	}

	{
		const unsigned int N = dataSize;
		double loss = 0.0;
		for (unsigned int k = 0u; k < realOut; ++k)
		{
			const float pred = tensorQnn.outReadout[k];
			const float expv = (yData && (k < ySize)) ? yData[k] : 0.0f;
			if (!is_finite(expv))
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_QNN: non-finite expected value detected (NaN/Inf)");
				running = false;
				return;
			}
			if (!is_finite(pred))
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_QNN: non-finite prediction detected (NaN/Inf)");
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
				const double denom = (N > 0u && realOut > 0u) ? static_cast<double>(N) * static_cast<double>(realOut) : 1.0;
				overallTotalError += static_cast<float>(GMath::outputNodeCost(expv, pred, static_cast<float>(denom), costFx));
			}
		}

		if (useSoftmax && N > 0u)
			overallTotalError += static_cast<float>(loss / static_cast<double>(N));
		if (!is_finite(overallTotalError))
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_QNN: non-finite loss aggregate detected (NaN/Inf)");
			running = false;
			return;
		}

		if ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL))
		{
			if (realOut > 1u)
			{
				const int expIdx = GMath::argmax(yData, realOut);
				const int predIdx = GMath::argmax(&tensorQnn.outReadout[0], realOut);
				confusionMatrix.addResultDirect(static_cast<unsigned int>(expIdx), static_cast<unsigned int>(predIdx));
				++clsTotal;
				if (expIdx == predIdx)
					++clsCorrect;
			}
			else
			{
				const float expv = (yData && ySize > 0u) ? yData[0] : 0.0f;
				const float pred = tensorQnn.outReadout[0];
				const unsigned int expCls = (expv >= 0.5f) ? 1u : 0u;
				const unsigned int predCls = (pred >= 0.5f) ? 1u : 0u;
				confusionMatrix.addResultDirect(expCls, predCls);
				++clsTotal;
				if (expCls == predCls)
					++clsCorrect;
			}
		}
	}

	if (!isTrain)
		return;

	// === Backpropagation in H ===
	{
		const int outActFx = skeleton->getActivationType(lastTransition);
		const float outActParam = skeleton->getActivationParam(lastTransition);

		// Output deltas (quaternion-valued, populated on active readout lanes).
		for (unsigned int m = 0u; m < tensorQnn.delta[lastLayer].size(); ++m)
			tensorQnn.delta[lastLayer][m] = Quaternion::zero();

		for (unsigned int k = 0u; k < realOut; ++k)
		{
			const float pred = tensorQnn.outReadout[k];
			const float expv = (yData && (k < ySize)) ? yData[k] : 0.0f;
			float dL_dz = 0.0f;
			if (useSoftmax)
			{
				dL_dz = clipf_maybe(pred - expv, gradClip);
			}
			else if ((costFx == GMath::CLASSIFICATION) && (outActFx == GMath::SIGMOID || outActFx == GMath::SIGMOIDP))
			{
				dL_dz = clipf_maybe(pred - expv, gradClip);
			}
			else
			{
				const float dCost_dA = GMath::costErrDer(expv, pred, costFx);
				const float dA_dZ = GMath::activationErrDer(pred, outActFx, outActParam);
				dL_dz = clipf_maybe(dCost_dA * dA_dZ, gradClip);
			}

			if (!is_finite(dL_dz))
			{
				lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_QNN: non-finite output delta detected (NaN/Inf)");
				running = false;
				return;
			}

			if (splitReadout)
				tensorQnn.delta[lastLayer][k / 4u].setComponent(k % 4u, dL_dz);
			else
				tensorQnn.delta[lastLayer][k].r = dL_dz;
		}

		// Hidden deltas: dL/da_n^(l) = sum_m (W_{m,n}^(l))* (x) delta_m^(l+1),
		// then component-wise split activation derivative.
		for (int li = static_cast<int>(lastLayer) - 1; li >= 1; --li)
		{
			const unsigned int l = static_cast<unsigned int>(li);
			const TensorQNNState::Transition& nextTr = tensorQnn.T[l];
			const int actFx = skeleton->getActivationType(l - 1u);
			const float actParam = skeleton->getActivationParam(l - 1u);
			const float sc = (l < keepScales.size()) ? keepScales[l] : 1.0f;
			const float invSc = (sc != 1.0f) ? (1.0f / sc) : 1.0f;

			for (unsigned int n = 0u; n < tensorQnn.qSizes[l]; ++n)
			{
				if (l < keepMasks.size() && n < keepMasks[l].size() && keepMasks[l][n] == 0u)
				{
					tensorQnn.delta[l][n] = Quaternion::zero();
					continue;
				}

				Quaternion gradA = Quaternion::zero();
				for (unsigned int m = 0u; m < nextTr.qOut; ++m)
				{
					const Quaternion& w = nextTr.W[static_cast<size_t>(m) * static_cast<size_t>(nextTr.qIn) + n];
					gradA += Quaternion::mulConjLeft(w, tensorQnn.delta[l + 1u][m]);
				}

				const Quaternion aUnscaled = (sc != 1.0f) ? (tensorQnn.a[l][n] * invSc) : tensorQnn.a[l][n];
				const Quaternion d(
				    clipf_maybe(gradA.r * sc * GMath::activationErrDer(aUnscaled.r, actFx, actParam), gradClip),
				    clipf_maybe(gradA.i * sc * GMath::activationErrDer(aUnscaled.i, actFx, actParam), gradClip),
				    clipf_maybe(gradA.j * sc * GMath::activationErrDer(aUnscaled.j, actFx, actParam), gradClip),
				    clipf_maybe(gradA.k * sc * GMath::activationErrDer(aUnscaled.k, actFx, actParam), gradClip));
				if (!d.isFinite())
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_QNN: non-finite hidden delta detected (NaN/Inf)");
					running = false;
					return;
				}
				tensorQnn.delta[l][n] = d;
			}
		}

		// Input-layer activation gradient dL/da^(0) (pre-dropout coordinates).
		{
			const TensorQNNState::Transition& firstTr = tensorQnn.T[0];
			const float sc0 = (!keepScales.empty()) ? keepScales[0] : 1.0f;
			for (unsigned int n = 0u; n < tensorQnn.qSizes[0]; ++n)
			{
				if (!keepMasks[0].empty() && n < keepMasks[0].size() && keepMasks[0][n] == 0u)
				{
					tensorQnn.delta[0][n] = Quaternion::zero();
					continue;
				}
				Quaternion gradIn = Quaternion::zero();
				for (unsigned int m = 0u; m < firstTr.qOut; ++m)
				{
					const Quaternion& w = firstTr.W[static_cast<size_t>(m) * static_cast<size_t>(firstTr.qIn) + n];
					gradIn += Quaternion::mulConjLeft(w, tensorQnn.delta[1][m]);
				}
				if (sc0 != 1.0f)
					gradIn *= sc0;
				tensorQnn.delta[0][n] = gradIn;
			}
		}

		// Accumulate minibatch gradients:
		//   dL/dW_{m,n}^(t) += delta_m^(t+1) (x) (a_n^(t))*
		//   dL/db_m^(t)     += delta_m^(t+1)
		for (unsigned int t = 0u; t < tensorQnn.T.size(); ++t)
		{
			TensorQNNState::Transition& tr = tensorQnn.T[t];
			for (unsigned int m = 0u; m < tr.qOut; ++m)
			{
				const Quaternion& d = tensorQnn.delta[t + 1u][m];
				if (m < tr.gBias.size())
					tr.gBias[m] += d;

				const size_t rowOff = static_cast<size_t>(m) * static_cast<size_t>(tr.qIn);
				for (unsigned int n = 0u; n < tr.qIn; ++n)
					tr.gW[rowOff + n] += Quaternion::mulConjRight(d, tensorQnn.a[t][n]);
			}
		}

		++tensorQnn.batchCount;

		// End-of-batch detection.
		const unsigned int trainSize = di ? di->getTrainSize() : 0u;
		const int effectiveMiniBatchSize = (minibatchSize > 0) ? minibatchSize : static_cast<int>(trainSize);
		const bool isLastSample = (trainSize > 0u) && (inputRowCounter + 1u >= trainSize);
		const bool isBatchEnd =
		    (effectiveMiniBatchSize <= 1) ||
		    (((inputRowCounter + 1u) % static_cast<unsigned int>(effectiveMiniBatchSize)) == 0u) ||
		    isLastSample;

		if (isBatchEnd && tensorQnn.batchCount > 0u)
		{
			const float invBatch = 1.0f / static_cast<float>(tensorQnn.batchCount);
			const bool useAdamW = (trainingConfig.optimizer.type == OptimizerConfig::ADAMW);

			// Optional global grad-norm clipping.
			float gradNorm = 0.0f;
			float gradScale = 1.0f;
			const float clipNorm = trainingConfig.globalGradClipNorm;
			if (clipNorm > 0.0f)
			{
				double sumsq = 0.0;
				for (unsigned int t = 0u; t < tensorQnn.T.size(); ++t)
				{
					const TensorQNNState::Transition& tr = tensorQnn.T[t];
					const float wd1 = skeleton->getWeightDecay1(t);
					const float wd2 = skeleton->getWeightDecay2(t);
					for (size_t idx = 0u; idx < tr.W.size(); ++idx)
						accumulate_q_grad_norm_sq(tr.W[idx], tr.gW[idx], invBatch, wd1, wd2, !useAdamW, sumsq);
					for (size_t m = 0u; m < tr.gBias.size(); ++m)
						accumulate_q_grad_norm_sq(tr.bias[m], tr.gBias[m], invBatch, 0.0f, 0.0f, false, sumsq);
				}
				if (!is_finite_double(sumsq))
				{
					lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_QNN: non-finite grad-norm accumulation detected (NaN/Inf)");
					running = false;
					return;
				}
				gradNorm = static_cast<float>(sqrt(sumsq));
				const float eps = 1e-12f;
				if (gradNorm > clipNorm)
					gradScale = clipNorm / (gradNorm + eps);
			}
			lastGradNorm = gradNorm;
			lastGradNormScale = gradScale;

			if (useAdamW)
			{
				++tensorQnn.optimizerStep;
				const float beta1 = trainingConfig.optimizer.adamBeta1;
				const float beta2 = trainingConfig.optimizer.adamBeta2;
				const float epsAdam = trainingConfig.optimizer.adamEps;
				const float stepF = static_cast<float>(tensorQnn.optimizerStep);
				const float bc1 = trainingConfig.optimizer.adamBiasCorrection ? (1.0f - powf(beta1, stepF)) : 1.0f;
				const float bc2 = trainingConfig.optimizer.adamBiasCorrection ? (1.0f - powf(beta2, stepF)) : 1.0f;

				for (unsigned int t = 0u; t < tensorQnn.T.size(); ++t)
				{
					TensorQNNState::Transition& tr = tensorQnn.T[t];
					const float lr = skeleton->getLearningRate(t) * lrScheduleMultiplier;
					const float wd1 = skeleton->getWeightDecay1(t);
					const float wd2 = skeleton->getWeightDecay2(t);

					for (size_t idx = 0u; idx < tr.W.size(); ++idx)
					{
						if (!adamw_update_quaternion(tr.W[idx], tr.vW[idx], tr.v2W[idx], tr.gW[idx],
						                             invBatch, gradScale, lr, wd1, wd2, true,
						                             beta1, beta2, bc1, bc2, epsAdam))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_QNN: non-finite AdamW weight update detected (NaN/Inf)");
							running = false;
							return;
						}
						tr.gW[idx] = Quaternion::zero();
					}

					for (size_t m = 0u; m < tr.bias.size(); ++m)
					{
						if (!adamw_update_quaternion(tr.bias[m], tr.vBias[m], tr.v2Bias[m], tr.gBias[m],
						                             invBatch, gradScale, lr, 0.0f, 0.0f, false,
						                             beta1, beta2, bc1, bc2, epsAdam))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_QNN: non-finite AdamW bias update detected (NaN/Inf)");
							running = false;
							return;
						}
						tr.gBias[m] = Quaternion::zero();
					}
				}
			}
			else
			{
				for (unsigned int t = 0u; t < tensorQnn.T.size(); ++t)
				{
					TensorQNNState::Transition& tr = tensorQnn.T[t];
					const float lr = skeleton->getLearningRate(t) * lrScheduleMultiplier;
					const float mf = skeleton->getMomentumFactor(t);
					const float wd1 = skeleton->getWeightDecay1(t);
					const float wd2 = skeleton->getWeightDecay2(t);

					for (size_t idx = 0u; idx < tr.W.size(); ++idx)
					{
						if (!sgd_update_quaternion(tr.W[idx], tr.vW[idx], tr.gW[idx],
						                           invBatch, gradScale, lr, mf, wd1, wd2, true))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_QNN: non-finite SGD weight update detected (NaN/Inf)");
							running = false;
							return;
						}
						tr.gW[idx] = Quaternion::zero();
					}

					for (size_t m = 0u; m < tr.bias.size(); ++m)
					{
						if (!sgd_update_quaternion(tr.bias[m], tr.vBias[m], tr.gBias[m],
						                           invBatch, gradScale, lr, mf, 0.0f, 0.0f, false))
						{
							lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_QNN: non-finite SGD bias update detected (NaN/Inf)");
							running = false;
							return;
						}
						tr.gBias[m] = Quaternion::zero();
					}
				}
			}

			tensorQnn.batchCount = 0u;
		}

		// Autotuning record parity with DFF path.
		{
			const float learningRate = skeleton->getLearningRate(lastTransition) * lrScheduleMultiplier;
			shmea::GList nbRow;
			nbRow.addFloat(overallTotalAccuracy);
			nbRow.addFloat(learningRate);
			nbRecord.addRow(nbRow);
		}
	}
}
