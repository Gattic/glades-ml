// Net-type-specific SGD: CNN tensor path (split out of network.cpp)
//
// Architecture: im2col + SGEMM convolution, optional BN, ReLU, optional MaxPool,
// followed by DFF-style FC layers with softmax + cross-entropy loss.
// Data layout: channel-first (NCHW) throughout.
#include "network.h"
#include "sgd_utils.h"

#include "Backend/Database/GLogger.h"

#include "../DataObjects/DataInput.h"
#include "../GMath/gmath.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <vector>

using namespace glades;

// ============================================================
// CPU im2col / col2im
// ============================================================
namespace {

static void im2col_cpu(const float* input,
                       unsigned int inC, unsigned int inH, unsigned int inW,
                       unsigned int kH, unsigned int kW,
                       unsigned int strideH, unsigned int strideW,
                       unsigned int padH, unsigned int padW,
                       unsigned int outH, unsigned int outW,
                       float* output)
{
	const unsigned int K = inC * kH * kW;
	for (unsigned int oh = 0; oh < outH; ++oh)
	{
		for (unsigned int ow = 0; ow < outW; ++ow)
		{
			const unsigned int row = oh * outW + ow;
			for (unsigned int c = 0; c < inC; ++c)
			{
				for (unsigned int kh = 0; kh < kH; ++kh)
				{
					for (unsigned int kww = 0; kww < kW; ++kww)
					{
						const int ih = static_cast<int>(oh * strideH) - static_cast<int>(padH) + static_cast<int>(kh);
						const int iw = static_cast<int>(ow * strideW) - static_cast<int>(padW) + static_cast<int>(kww);
						float val = 0.0f;
						if (ih >= 0 && ih < static_cast<int>(inH) && iw >= 0 && iw < static_cast<int>(inW))
							val = input[c * inH * inW + static_cast<unsigned int>(ih) * inW + static_cast<unsigned int>(iw)];
						output[row * K + c * kH * kW + kh * kW + kww] = val;
					}
				}
			}
		}
	}
}

static void col2im_cpu(const float* cols,
                       unsigned int inC, unsigned int inH, unsigned int inW,
                       unsigned int kH, unsigned int kW,
                       unsigned int strideH, unsigned int strideW,
                       unsigned int padH, unsigned int padW,
                       unsigned int outH, unsigned int outW,
                       float* output)
{
	// Zero output first.
	std::memset(output, 0, static_cast<size_t>(inC) * inH * inW * sizeof(float));
	const unsigned int K = inC * kH * kW;
	for (unsigned int oh = 0; oh < outH; ++oh)
	{
		for (unsigned int ow = 0; ow < outW; ++ow)
		{
			const unsigned int row = oh * outW + ow;
			for (unsigned int c = 0; c < inC; ++c)
			{
				for (unsigned int kh = 0; kh < kH; ++kh)
				{
					for (unsigned int kww = 0; kww < kW; ++kww)
					{
						const int ih = static_cast<int>(oh * strideH) - static_cast<int>(padH) + static_cast<int>(kh);
						const int iw = static_cast<int>(ow * strideW) - static_cast<int>(padW) + static_cast<int>(kww);
						if (ih >= 0 && ih < static_cast<int>(inH) && iw >= 0 && iw < static_cast<int>(inW))
							output[c * inH * inW + static_cast<unsigned int>(ih) * inW + static_cast<unsigned int>(iw)] +=
							    cols[row * K + c * kH * kW + kh * kW + kww];
					}
				}
			}
		}
	}
}

// Row-major GEMM: C[M,N] += A[M,K] * B[K,N].
// Loop order M,K,N: inner loop scans B and C rows contiguously (stride 1).
// Inner loop unrolled 4-wide to reduce loop overhead.
static void sgemm_cpu(const float* A, const float* B, float* C,
                      unsigned int M, unsigned int K, unsigned int N)
{
	for (unsigned int m = 0; m < M; ++m)
	{
		for (unsigned int k = 0; k < K; ++k)
		{
			const float a = A[m * K + k];
			const float* Brow = B + k * N;
			float* Crow = C + m * N;
			unsigned int n = 0;
			for (; n + 3 < N; n += 4)
			{
				Crow[n]     += a * Brow[n];
				Crow[n + 1] += a * Brow[n + 1];
				Crow[n + 2] += a * Brow[n + 2];
				Crow[n + 3] += a * Brow[n + 3];
			}
			for (; n < N; ++n)
				Crow[n] += a * Brow[n];
		}
	}
}

// C[M,N] += A[M,K] * B^T[N,K].  B stored as [N,K].
// Inner dot-product loop unrolled 4-wide to reduce loop overhead.
static void sgemm_abt_cpu(const float* A, const float* B, float* C,
                          unsigned int M, unsigned int K, unsigned int N)
{
	for (unsigned int m = 0; m < M; ++m)
	{
		const float* Arow = A + m * K;
		for (unsigned int n = 0; n < N; ++n)
		{
			const float* Brow = B + n * K;
			float sum = 0.0f;
			unsigned int k = 0;
			for (; k + 3 < K; k += 4)
			{
				sum += Arow[k]     * Brow[k]
				     + Arow[k + 1] * Brow[k + 1]
				     + Arow[k + 2] * Brow[k + 2]
				     + Arow[k + 3] * Brow[k + 3];
			}
			for (; k < K; ++k)
				sum += Arow[k] * Brow[k];
			C[m * N + n] += sum;
		}
	}
}

// C[M,N] += A^T[K,M] * B[K,N].  A stored as [K,M].
// Loop order K,M,N: inner loop scans B and C rows contiguously (stride 1).
// Inner loop unrolled 4-wide to reduce loop overhead.
static void sgemm_atb_cpu(const float* A, const float* B, float* C,
                          unsigned int K, unsigned int M, unsigned int N)
{
	for (unsigned int k = 0; k < K; ++k)
	{
		const float* Brow = B + k * N;
		for (unsigned int m = 0; m < M; ++m)
		{
			const float a = A[k * M + m];
			float* Crow = C + m * N;
			unsigned int n = 0;
			for (; n + 3 < N; n += 4)
			{
				Crow[n]     += a * Brow[n];
				Crow[n + 1] += a * Brow[n + 1];
				Crow[n + 2] += a * Brow[n + 2];
				Crow[n + 3] += a * Brow[n + 3];
			}
			for (; n < N; ++n)
				Crow[n] += a * Brow[n];
		}
	}
}

static void maxpool_forward_cpu(const float* input,
                                unsigned int C, unsigned int inH, unsigned int inW,
                                unsigned int poolH, unsigned int poolW,
                                unsigned int poolStrideH, unsigned int poolStrideW,
                                unsigned int outH, unsigned int outW,
                                float* output, int* argmax)
{
	for (unsigned int c = 0; c < C; ++c)
	{
		for (unsigned int oh = 0; oh < outH; ++oh)
		{
			for (unsigned int ow = 0; ow < outW; ++ow)
			{
				const unsigned int outIdx = c * outH * outW + oh * outW + ow;
				float maxVal = -1e30f;
				int maxIdx = 0;
				for (unsigned int ph = 0; ph < poolH; ++ph)
				{
					for (unsigned int pw = 0; pw < poolW; ++pw)
					{
						const unsigned int ih = oh * poolStrideH + ph;
						const unsigned int iw = ow * poolStrideW + pw;
						if (ih < inH && iw < inW)
						{
							const unsigned int srcIdx = c * inH * inW + ih * inW + iw;
							if (input[srcIdx] > maxVal)
							{
								maxVal = input[srcIdx];
								maxIdx = static_cast<int>(srcIdx);
							}
						}
					}
				}
				output[outIdx] = maxVal;
				argmax[outIdx] = maxIdx;
			}
		}
	}
}

static void maxpool_backward_cpu(const float* dOutput, const int* argmax,
                                 unsigned int C, unsigned int inH, unsigned int inW,
                                 unsigned int outH, unsigned int outW,
                                 float* dInput)
{
	std::memset(dInput, 0, static_cast<size_t>(C) * inH * inW * sizeof(float));
	const unsigned int total = C * outH * outW;
	for (unsigned int i = 0; i < total; ++i)
		dInput[argmax[i]] += dOutput[i];
}

static void batchnorm_forward_train_cpu(const float* input,
                                        unsigned int C, unsigned int N,
                                        const float* gamma, const float* beta,
                                        float eps,
                                        float* output,
                                        float* mean, float* invStd, float* normalized)
{
	for (unsigned int c = 0; c < C; ++c)
	{
		const float* x = input + c * N;
		float* y = output + c * N;
		float* norm = normalized + c * N;

		// Mean
		double sum = 0.0;
		for (unsigned int i = 0; i < N; ++i)
			sum += static_cast<double>(x[i]);
		const float m = static_cast<float>(sum / static_cast<double>(N));
		mean[c] = m;

		// Variance
		double varSum = 0.0;
		for (unsigned int i = 0; i < N; ++i)
		{
			const double d = static_cast<double>(x[i]) - static_cast<double>(m);
			varSum += d * d;
		}
		const float var = static_cast<float>(varSum / static_cast<double>(N));
		const float is = 1.0f / sqrtf(var + eps);
		invStd[c] = is;

		// Normalize + scale/shift
		const float g = gamma[c];
		const float b = beta[c];
		for (unsigned int i = 0; i < N; ++i)
		{
			const float n = (x[i] - m) * is;
			norm[i] = n;
			y[i] = g * n + b;
		}
	}
}

static void batchnorm_backward_cpu(const float* dOutput,
                                   const float* normalized,
                                   const float* gamma,
                                   const float* invStd,
                                   unsigned int C, unsigned int N,
                                   float* dInput,
                                   float* dgamma, float* dbeta)
{
	for (unsigned int c = 0; c < C; ++c)
	{
		const float* dy = dOutput + c * N;
		const float* xn = normalized + c * N;
		float* dx = dInput + c * N;

		float dg = 0.0f, db = 0.0f;
		for (unsigned int i = 0; i < N; ++i)
		{
			dg += dy[i] * xn[i];
			db += dy[i];
		}
		dgamma[c] += dg;
		dbeta[c] += db;

		const float g = gamma[c];
		const float is = invStd[c];
		const float invN = 1.0f / static_cast<float>(N);
		for (unsigned int i = 0; i < N; ++i)
			dx[i] = g * is * invN * (static_cast<float>(N) * dy[i] - db - xn[i] * dg);
	}
}

} // anonymous namespace

// ============================================================
// SGDHelper_CNN
// ============================================================
void glades::NNetwork::SGDHelper_CNN(unsigned int inputRowCounter, int runType)
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
	if (!tensorCnn.initialized)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_STATE, "SGDHelper_CNN: tensor state is not initialized");
		running = false;
		return;
	}

	const TensorCNNState& cs = tensorCnn;
	const CNNConfig& cfg = trainingConfig.cnn;
	const unsigned int numConv = static_cast<unsigned int>(cs.convLayers.size());
	const unsigned int numFC = static_cast<unsigned int>(cs.fcLayers.size());
	const unsigned int outSize = (numFC > 0u) ? cs.fcLayers[numFC - 1u].out : 0u;

	// Get input data.
	const float* xData = NULL;
	unsigned int xSize = 0u;
	if (di)
	{
		if (isTrain)
			di->getTrainRowView(inputRowCounter, xData, xSize);
		else
			di->getTestRowView(inputRowCounter, xData, xSize);
	}
	if (!xData || xSize == 0u)
	{
		lastStatus = NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "SGDHelper_CNN: empty input row");
		running = false;
		return;
	}

	// Ensure scratch buffers.
	if (cnnScratch.convScratch.size() != numConv)
	{
		cnnScratch.convScratch.resize(numConv);
		for (unsigned int l = 0; l < numConv; ++l)
		{
			const TensorCNNState::ConvSpatialInfo& sp = cs.spatialInfo[l];
			CNNScratch::ConvLayerScratch& sc = cnnScratch.convScratch[l];
			const size_t N = static_cast<size_t>(sp.outH) * sp.outW;

			sc.im2col.assign(static_cast<size_t>(sp.im2colRows) * sp.im2colCols, 0.0f);
			sc.convOut.assign(static_cast<size_t>(sp.outC) * N, 0.0f);
			if (sp.useBatchNorm)
			{
				sc.bnOut.assign(static_cast<size_t>(sp.outC) * N, 0.0f);
				sc.bnMean.assign(sp.outC, 0.0f);
				sc.bnInvStd.assign(sp.outC, 0.0f);
				sc.bnNorm.assign(static_cast<size_t>(sp.outC) * N, 0.0f);
			}
			sc.actOut.assign(static_cast<size_t>(sp.outC) * N, 0.0f);
			if (sp.useMaxPool)
			{
				const size_t poolN = static_cast<size_t>(sp.outC) * sp.poolOutH * sp.poolOutW;
				sc.poolOut.assign(poolN, 0.0f);
				sc.poolArgmax.assign(poolN, 0);
			}
			// Backward
			if (sp.useMaxPool)
				sc.dPoolOut.assign(static_cast<size_t>(sp.outC) * sp.poolOutH * sp.poolOutW, 0.0f);
			sc.dActOut.assign(static_cast<size_t>(sp.outC) * N, 0.0f);
			sc.dConvOut.assign(static_cast<size_t>(sp.outC) * N, 0.0f);
			sc.dIm2col.assign(static_cast<size_t>(sp.im2colRows) * sp.im2colCols, 0.0f);
		}
	}

	// Ensure FC scratch.
	if (cnnScratch.fcA.size() != numFC + 1u)
	{
		cnnScratch.fcA.resize(numFC + 1u);
		cnnScratch.fcDelta.resize(numFC + 1u);
		cnnScratch.fcA[0].assign(cs.flattenedSize, 0.0f);
		cnnScratch.fcDelta[0].clear(); // no delta for input
		for (unsigned int t = 0; t < numFC; ++t)
		{
			cnnScratch.fcA[t + 1u].assign(cs.fcLayers[t].out, 0.0f);
			cnnScratch.fcDelta[t + 1u].assign(cs.fcLayers[t].out, 0.0f);
		}
		cnnScratch.logits.assign(outSize, 0.0f);
		cnnScratch.probs.assign(outSize, 0.0f);
		cnnScratch.dFlatten.assign(cs.flattenedSize, 0.0f);

		// Pre-allocate convInputGrad to the max input volume across all conv layers.
		size_t maxInputVol = 0;
		for (unsigned int l = 0; l < numConv; ++l)
		{
			const size_t vol = static_cast<size_t>(cs.spatialInfo[l].inC) * cs.spatialInfo[l].inH * cs.spatialInfo[l].inW;
			if (vol > maxInputVol) maxInputVol = vol;
		}
		if (maxInputVol > 0)
			cnnScratch.convInputGrad.assign(maxInputVol, 0.0f);
	}

	// ==============================
	// FORWARD PASS: Conv layers
	// ==============================
	// Input image is xData in NCHW format [C, H, W].
	const float* curInput = xData;

	for (unsigned int l = 0; l < numConv; ++l)
	{
		const TensorCNNState::ConvSpatialInfo& sp = cs.spatialInfo[l];
		const TensorCNNState::ConvLayer& cl = cs.convLayers[l];
		CNNScratch::ConvLayerScratch& sc = cnnScratch.convScratch[l];
		const unsigned int N = sp.outH * sp.outW; // spatial positions
		const unsigned int K = sp.im2colCols;      // inC * kH * kW

		// 1. im2col
		im2col_cpu(curInput, sp.inC, sp.inH, sp.inW,
		           sp.kH, sp.kW, sp.strideH, sp.strideW, sp.padH, sp.padW,
		           sp.outH, sp.outW, &sc.im2col[0]);

		// 2. GEMM: convOut[outC, N] = bias + W[outC, K] * im2col^T[K, N]
		//    Pre-fill convOut with broadcast bias, then GEMM accumulates on top.
		for (unsigned int c = 0; c < sp.outC; ++c)
		{
			const float b = cl.bias[c];
			for (unsigned int n = 0; n < N; ++n)
				sc.convOut[c * N + n] = b;
		}
		sgemm_abt_cpu(&cl.W[0], &sc.im2col[0], &sc.convOut[0], sp.outC, K, N);

		// 4. BatchNorm (if enabled)
		float* preAct = &sc.convOut[0];
		if (sp.useBatchNorm)
		{
			if (isTrain)
			{
				batchnorm_forward_train_cpu(preAct, sp.outC, N,
				                            &cl.bnGamma[0], &cl.bnBeta[0],
				                            cfg.batchNormEps,
				                            &sc.bnOut[0], &sc.bnMean[0], &sc.bnInvStd[0], &sc.bnNorm[0]);

				// Update running statistics (EMA).
				TensorCNNState::ConvLayer& clMut = tensorCnn.convLayers[l];
				const float mom = cfg.batchNormMomentum;
				for (unsigned int c = 0; c < sp.outC; ++c)
				{
					clMut.bnRunMean[c] = (1.0f - mom) * clMut.bnRunMean[c] + mom * sc.bnMean[c];
					clMut.bnRunVar[c] = (1.0f - mom) * clMut.bnRunVar[c] + mom * (1.0f / (sc.bnInvStd[c] * sc.bnInvStd[c]) - cfg.batchNormEps);
				}
			}
			else
			{
				// Inference: use running mean/var.
				for (unsigned int c = 0; c < sp.outC; ++c)
				{
					const float invStd = 1.0f / sqrtf(cl.bnRunVar[c] + cfg.batchNormEps);
					const float g = cl.bnGamma[c];
					const float b = cl.bnBeta[c];
					for (unsigned int n = 0; n < N; ++n)
						sc.bnOut[c * N + n] = g * (preAct[c * N + n] - cl.bnRunMean[c]) * invStd + b;
				}
			}
			preAct = &sc.bnOut[0];
		}

		// 5. ReLU activation
		for (size_t i = 0; i < static_cast<size_t>(sp.outC) * N; ++i)
			sc.actOut[i] = (preAct[i] > 0.0f) ? preAct[i] : 0.0f;

		// 6. MaxPool (if enabled)
		if (sp.useMaxPool)
		{
			maxpool_forward_cpu(&sc.actOut[0], sp.outC, sp.outH, sp.outW,
			                    sp.poolH, sp.poolW, sp.poolStrideH, sp.poolStrideW,
			                    sp.poolOutH, sp.poolOutW,
			                    &sc.poolOut[0], &sc.poolArgmax[0]);
			curInput = &sc.poolOut[0];
		}
		else
		{
			curInput = &sc.actOut[0];
		}
	}

	// ==============================
	// FORWARD PASS: Flatten + FC layers
	// ==============================
	// Copy flattened conv output into FC input.
	std::memcpy(&cnnScratch.fcA[0][0], curInput, cs.flattenedSize * sizeof(float));

	const bool useSoftmax = (outSize > 1u);
	const int costFx = skeleton->getOutputType();

	for (unsigned int t = 0; t < numFC; ++t)
	{
		const TensorCNNState::FCTransition& fc = cs.fcLayers[t];
		const bool isLastFC = (t == numFC - 1u);

		std::memset(&cnnScratch.fcA[t + 1u][0], 0, cnnScratch.fcA[t + 1u].size() * sizeof(float));

		for (unsigned int j = 0; j < fc.out; ++j)
		{
			float z = fc.bias[j];
			const size_t rowOff = static_cast<size_t>(j) * fc.in;
			for (unsigned int i = 0; i < fc.in; ++i)
				z += fc.W[rowOff + i] * cnnScratch.fcA[t][i];

			if (isLastFC && useSoftmax)
			{
				cnnScratch.logits[j] = z;
			}
			else
			{
				// ReLU for hidden FC layers.
				cnnScratch.fcA[t + 1u][j] = (z > 0.0f) ? z : 0.0f;
			}
		}

		if (isLastFC && useSoftmax)
		{
			softmax_stable(cnnScratch.logits, cnnScratch.probs);
			for (unsigned int j = 0; j < fc.out; ++j)
				cnnScratch.fcA[t + 1u][j] = cnnScratch.probs[j];
		}
	}

	// ==============================
	// Loss + metrics
	// ==============================
	if (!isTrain)
		results.clear();

	const float* yData = NULL;
	unsigned int ySize = 0u;
	if (di)
	{
		if (isTrain)
			di->getTrainExpectedRowView(inputRowCounter, yData, ySize);
		else
			di->getTestExpectedRowView(inputRowCounter, yData, ySize);
	}

	const float* pred = &cnnScratch.fcA[numFC][0];
	double loss = 0.0;
	for (unsigned int k = 0; k < outSize; ++k)
	{
		const float p = pred[k];
		const float expv = (yData && (k < ySize)) ? yData[k] : 0.0f;
		if (!isTrain)
		{
			results.addFloat(expv);
			results.addFloat(p);
		}

		if (costFx == GMath::REGRESSION)
		{
			const double diff = static_cast<double>(p) - static_cast<double>(expv);
			regSSE += diff * diff;
			regSAE += fabs(diff);
			regSumY += static_cast<double>(expv);
			regSumY2 += static_cast<double>(expv) * static_cast<double>(expv);
			++regCount;
		}
		else if (useSoftmax)
		{
			const float pp = sgd_detail::clamp_prob01(p);
			loss += -static_cast<double>(expv) * log(static_cast<double>(pp));
		}
	}

	if (useSoftmax && dataSize > 0u)
		overallTotalError += static_cast<float>(loss / static_cast<double>(dataSize));

	if ((costFx == GMath::CLASSIFICATION) || (costFx == GMath::KL))
	{
		const int expIdx = GMath::argmax(yData, outSize);
		const int predIdx = GMath::argmax(pred, outSize);
		confusionMatrix.addResultDirect(static_cast<unsigned int>(expIdx), static_cast<unsigned int>(predIdx));
	}

	// ==============================
	// BACKWARD PASS (train only)
	// ==============================
	if (!isTrain)
		return;

	// --- FC backward ---
	// Output deltas: softmax + CE => d = p - y
	for (unsigned int k = 0; k < outSize; ++k)
	{
		const float p = pred[k];
		const float expv = (yData && (k < ySize)) ? yData[k] : 0.0f;
		cnnScratch.fcDelta[numFC][k] = clipf_maybe(p - expv, gradClip);
	}

	// Hidden FC deltas (backwards)
	for (int li = static_cast<int>(numFC) - 1; li >= 1; --li)
	{
		const unsigned int l = static_cast<unsigned int>(li);
		const TensorCNNState::FCTransition& nextFC = cs.fcLayers[l];

		for (unsigned int i = 0; i < nextFC.in; ++i)
		{
			float sum = 0.0f;
			for (unsigned int j = 0; j < nextFC.out; ++j)
				sum += cnnScratch.fcDelta[l + 1u][j] * nextFC.W[static_cast<size_t>(j) * nextFC.in + i];

			// ReLU derivative
			const float a = cnnScratch.fcA[l][i];
			cnnScratch.fcDelta[l][i] = clipf_maybe((a > 0.0f) ? sum : 0.0f, gradClip);
		}
	}

	// FC gradient accumulation
	for (unsigned int t = 0; t < numFC; ++t)
	{
		TensorCNNState::FCTransition& fc = tensorCnn.fcLayers[t];
		for (unsigned int j = 0; j < fc.out; ++j)
		{
			const float d = cnnScratch.fcDelta[t + 1u][j];
			fc.gBias[j] += d;
			const size_t rowOff = static_cast<size_t>(j) * fc.in;
			for (unsigned int i = 0; i < fc.in; ++i)
				fc.gW[rowOff + i] += d * cnnScratch.fcA[t][i];
		}
	}

	// --- Conv backward ---
	// Gradient from FC input layer = fcDelta[0] if numFC > 0, else output deltas.
	// But the first FC layer's delta is propagated through W[0]:
	// dFlatten[i] = sum_j delta[1][j] * W[0][j, i] * relu'(a[0][i])
	// However, fcDelta[0] was not computed above (we stop at li >= 1).
	// We need dFlatten = sum_j delta[1][j] * W[0][j, i] (no relu on flatten).
	// Actually, the flatten layer has no activation, so:
	std::memset(&cnnScratch.dFlatten[0], 0, cnnScratch.dFlatten.size() * sizeof(float));
	if (numFC > 0u)
	{
		const TensorCNNState::FCTransition& firstFC = cs.fcLayers[0];
		for (unsigned int i = 0; i < firstFC.in; ++i)
		{
			float sum = 0.0f;
			for (unsigned int j = 0; j < firstFC.out; ++j)
				sum += cnnScratch.fcDelta[1u][j] * firstFC.W[static_cast<size_t>(j) * firstFC.in + i];
			cnnScratch.dFlatten[i] = sum;
		}
	}

	// Propagate through conv layers (top to bottom).
	const float* dUpstream = &cnnScratch.dFlatten[0];

	for (int li = static_cast<int>(numConv) - 1; li >= 0; --li)
	{
		const unsigned int l = static_cast<unsigned int>(li);
		const TensorCNNState::ConvSpatialInfo& sp = cs.spatialInfo[l];
		TensorCNNState::ConvLayer& cl = tensorCnn.convLayers[l];
		CNNScratch::ConvLayerScratch& sc = cnnScratch.convScratch[l];
		const unsigned int N = sp.outH * sp.outW;
		const unsigned int K = sp.im2colCols;

		// 1. MaxPool backward
		if (sp.useMaxPool)
		{
			// dUpstream is [outC, poolOutH, poolOutW]
			// Copy into dPoolOut.
			std::memcpy(&sc.dPoolOut[0], dUpstream, static_cast<size_t>(sp.outC) * sp.poolOutH * sp.poolOutW * sizeof(float));
			// Scatter to actOut space.
			maxpool_backward_cpu(&sc.dPoolOut[0], &sc.poolArgmax[0],
			                     sp.outC, sp.outH, sp.outW,
			                     sp.poolOutH, sp.poolOutW,
			                     &sc.dActOut[0]);
		}
		else
		{
			// No pool: dUpstream is [outC, outH, outW] = [outC, N]
			std::memcpy(&sc.dActOut[0], dUpstream, static_cast<size_t>(sp.outC) * N * sizeof(float));
		}

		// 2. ReLU backward
		const float* preAct = sp.useBatchNorm ? &sc.bnOut[0] : &sc.convOut[0];
		for (size_t i = 0; i < static_cast<size_t>(sp.outC) * N; ++i)
			sc.dActOut[i] = (preAct[i] > 0.0f) ? sc.dActOut[i] : 0.0f;

		// 3. BatchNorm backward
		float* dConvInput = &sc.dActOut[0]; // gradient w.r.t. conv output (or BN input)
		if (sp.useBatchNorm)
		{
			batchnorm_backward_cpu(&sc.dActOut[0], &sc.bnNorm[0],
			                       &cl.bnGamma[0], &sc.bnInvStd[0],
			                       sp.outC, N,
			                       &sc.dConvOut[0],
			                       &cl.gBnGamma[0], &cl.gBnBeta[0]);
			dConvInput = &sc.dConvOut[0];
		}

		// 4. Weight gradient: gW[outC, K] += dConvOut[outC, N] * im2col[N, K]
		//    = dConvOut * im2col (standard GEMM, A=[outC,N], B=[N,K])
		sgemm_cpu(dConvInput, &sc.im2col[0], &cl.gW[0], sp.outC, N, K);

		// 5. Bias gradient: gBias[c] += sum over spatial of dConvOut[c, :]
		for (unsigned int c = 0; c < sp.outC; ++c)
		{
			float bsum = 0.0f;
			for (unsigned int n = 0; n < N; ++n)
				bsum += dConvInput[c * N + n];
			cl.gBias[c] += bsum;
		}

		// 6. Input gradient: dIm2col[N, K] = dConvOut^T[N, outC] * W[outC, K]
		//    = (dConvOut)^T * W = ATB where A=dConvOut[outC,N], B=W[outC,K]
		//    So dIm2col = A^T * B: A stored as [outC, N] => A^T is [N, outC]
		std::memset(&sc.dIm2col[0], 0, sc.dIm2col.size() * sizeof(float));
		sgemm_atb_cpu(dConvInput, &cl.W[0], &sc.dIm2col[0], sp.outC, N, K);

		// col2im to get dInput for previous layer
		if (l > 0u)
		{
			// convInputGrad is pre-allocated to max input volume; col2im_cpu zeros it internally.
			col2im_cpu(&sc.dIm2col[0], sp.inC, sp.inH, sp.inW,
			           sp.kH, sp.kW, sp.strideH, sp.strideW, sp.padH, sp.padW,
			           sp.outH, sp.outW, &cnnScratch.convInputGrad[0]);
			dUpstream = &cnnScratch.convInputGrad[0];
		}
		// For l==0, we don't need the input gradient (no further layers to propagate to).
	}

	// ==============================
	// Weight update
	// ==============================
	++tensorCnn.batchCount;

	const unsigned int trainSize = di ? di->getTrainSize() : 0;
	const int effectiveBatchSize = (minibatchSize > 0) ? minibatchSize : static_cast<int>(trainSize);
	const bool isLastSample = (trainSize > 0) && (inputRowCounter + 1 >= trainSize);
	const bool isBatchEnd =
	    (effectiveBatchSize <= 1) ||
	    (((inputRowCounter + 1) % static_cast<unsigned int>(effectiveBatchSize)) == 0) ||
	    isLastSample;

	if (!isBatchEnd || tensorCnn.batchCount == 0u)
		return;

	const float invBatch = 1.0f / static_cast<float>(tensorCnn.batchCount);
	const bool useAtlas = (trainingConfig.optimizer.type == OptimizerConfig::ATLAS);
	const bool useAdam = (trainingConfig.optimizer.type == OptimizerConfig::ADAMW);
	++tensorCnn.optimizerStep;
	const float baseLR = skeleton->getLearningRate(0) * lrScheduleMultiplier;
	const float mf = skeleton->getMomentumFactor(0);
	const float wd2 = skeleton->getWeightDecay2(0);

	// Global gradient-norm clipping (matching DFF path).
	float gradNorm = 0.0f;
	float gradScale = 1.0f;
	const float clipNorm = trainingConfig.globalGradClipNorm;
	if (clipNorm > 0.0f)
	{
		double sumsq = 0.0;
		for (unsigned int l = 0; l < numConv; ++l)
		{
			const TensorCNNState::ConvLayer& cl = tensorCnn.convLayers[l];
			for (size_t i = 0; i < cl.gW.size(); ++i)
			{
				float g = cl.gW[i] * invBatch + wd2 * cl.W[i];
				sumsq += static_cast<double>(g) * static_cast<double>(g);
			}
			for (unsigned int c = 0; c < cl.outC; ++c)
			{
				const double gb = static_cast<double>(cl.gBias[c] * invBatch);
				sumsq += gb * gb;
			}
			if (cs.spatialInfo[l].useBatchNorm)
			{
				for (unsigned int c = 0; c < cl.outC; ++c)
				{
					const double gg = static_cast<double>(cl.gBnGamma[c] * invBatch);
					const double gb = static_cast<double>(cl.gBnBeta[c] * invBatch);
					sumsq += gg * gg + gb * gb;
				}
			}
		}
		for (unsigned int t = 0; t < numFC; ++t)
		{
			const TensorCNNState::FCTransition& fc = tensorCnn.fcLayers[t];
			for (size_t i = 0; i < fc.gW.size(); ++i)
			{
				float g = fc.gW[i] * invBatch + wd2 * fc.W[i];
				sumsq += static_cast<double>(g) * static_cast<double>(g);
			}
			for (unsigned int j = 0; j < fc.out; ++j)
			{
				const double gb = static_cast<double>(fc.gBias[j] * invBatch);
				sumsq += gb * gb;
			}
		}

		if (!is_finite_double(sumsq))
		{
			lastStatus = NNetworkStatus(NNetworkStatus::INTERNAL_ERROR, "SGDHelper_CNN: non-finite grad-norm accumulation detected (NaN/Inf)");
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

	if (useAtlas)
	{
		const ATLASConfig& ac = trainingConfig.atlas;
		const float wd1 = skeleton->getWeightDecay1(0);

		// Update conv layers
		for (unsigned int l = 0; l < numConv; ++l)
		{
			TensorCNNState::ConvLayer& cl = tensorCnn.convLayers[l];
			const unsigned int convM = cl.outC;
			const unsigned int convN = cl.inC * cl.kH * cl.kW;

			// Initialize ATLAS state for this conv layer if needed
			if (!cl.atlasW.initialized && convM > 0 && convN > 0)
				atlas::initWeightState(cl.atlasW, convM, convN, ac.rank, ac.muMin, rngEngine, getLogger());

			if (cl.atlasW.initialized)
			{
				// ATLAS step for conv weight matrix
				atlas::applyStep(cl.atlasW,
					&cl.W[0], &cl.gW[0],
					convM, convN,
					invBatch, baseLR,
					wd1, wd2, gradScale,
					ac.beta, ac.muMin, ac.muMax,
					ac.eps, ac.tSub, ac.powerIters, ac.betaRefresh,
					rngEngine, getLogger());
			}

			// Bias update: standard SGD (no subspace projection for 1D vectors)
			for (unsigned int c = 0; c < cl.outC; ++c)
			{
				float g = cl.gBias[c] * invBatch * gradScale;
				cl.bias[c] -= baseLR * g;
				cl.gBias[c] = 0.0f;
			}

			// BatchNorm params: standard SGD (no subspace projection)
			if (cs.spatialInfo[l].useBatchNorm)
			{
				for (unsigned int c = 0; c < cl.outC; ++c)
				{
					{
						float g = cl.gBnGamma[c] * invBatch * gradScale;
						cl.bnGamma[c] -= baseLR * g;
						cl.gBnGamma[c] = 0.0f;
					}
					{
						float g = cl.gBnBeta[c] * invBatch * gradScale;
						cl.bnBeta[c] -= baseLR * g;
						cl.gBnBeta[c] = 0.0f;
					}
				}
			}
		}

		// Update FC layers
		for (unsigned int t = 0; t < numFC; ++t)
		{
			TensorCNNState::FCTransition& fc = tensorCnn.fcLayers[t];

			// Initialize ATLAS state for this FC layer if needed
			if (!fc.atlasW.initialized && fc.out > 0 && fc.in > 0)
				atlas::initWeightState(fc.atlasW, fc.out, fc.in, ac.rank, ac.muMin, rngEngine, getLogger());

			if (fc.atlasW.initialized)
			{
				// ATLAS step for FC weight matrix
				atlas::applyStep(fc.atlasW,
					&fc.W[0], &fc.gW[0],
					fc.out, fc.in,
					invBatch, baseLR,
					wd1, wd2, gradScale,
					ac.beta, ac.muMin, ac.muMax,
					ac.eps, ac.tSub, ac.powerIters, ac.betaRefresh,
					rngEngine, getLogger());
			}

			// Bias update: standard SGD (no subspace projection for 1D vectors)
			for (unsigned int j = 0; j < fc.out; ++j)
			{
				float g = fc.gBias[j] * invBatch * gradScale;
				fc.bias[j] -= baseLR * g;
				fc.gBias[j] = 0.0f;
			}
		}
	}
	else if (useAdam)
	{
		const float beta1 = trainingConfig.optimizer.adamBeta1;
		const float beta2 = trainingConfig.optimizer.adamBeta2;
		const float epsAdam = trainingConfig.optimizer.adamEps;
		const unsigned long long step = tensorCnn.optimizerStep;
		const float bc1 = 1.0f - powf(beta1, static_cast<float>(step));
		const float bc2 = 1.0f - powf(beta2, static_cast<float>(step));

		// Update conv layers
		for (unsigned int l = 0; l < numConv; ++l)
		{
			TensorCNNState::ConvLayer& cl = tensorCnn.convLayers[l];

			for (size_t i = 0; i < cl.W.size(); ++i)
			{
				float g = cl.gW[i] * invBatch * gradScale;
				cl.vW[i] = beta1 * cl.vW[i] + (1.0f - beta1) * g;
				cl.v2W[i] = beta2 * cl.v2W[i] + (1.0f - beta2) * g * g;
				const float mHat = cl.vW[i] / bc1;
				const float vHat = cl.v2W[i] / bc2;
				cl.W[i] -= baseLR * (mHat / (sqrtf(vHat) + epsAdam) + wd2 * cl.W[i]);
			}

			for (unsigned int c = 0; c < cl.outC; ++c)
			{
				float g = cl.gBias[c] * invBatch * gradScale;
				cl.vBias[c] = beta1 * cl.vBias[c] + (1.0f - beta1) * g;
				cl.v2Bias[c] = beta2 * cl.v2Bias[c] + (1.0f - beta2) * g * g;
				const float mHat = cl.vBias[c] / bc1;
				const float vHat = cl.v2Bias[c] / bc2;
				cl.bias[c] -= baseLR * mHat / (sqrtf(vHat) + epsAdam);
			}

			// BN params
			if (cs.spatialInfo[l].useBatchNorm)
			{
				for (unsigned int c = 0; c < cl.outC; ++c)
				{
					{
						float g = cl.gBnGamma[c] * invBatch * gradScale;
						cl.vBnGamma[c] = beta1 * cl.vBnGamma[c] + (1.0f - beta1) * g;
						cl.v2BnGamma[c] = beta2 * cl.v2BnGamma[c] + (1.0f - beta2) * g * g;
						const float mH = cl.vBnGamma[c] / bc1;
						const float vH = cl.v2BnGamma[c] / bc2;
						cl.bnGamma[c] -= baseLR * mH / (sqrtf(vH) + epsAdam);
					}
					{
						float g = cl.gBnBeta[c] * invBatch * gradScale;
						cl.vBnBeta[c] = beta1 * cl.vBnBeta[c] + (1.0f - beta1) * g;
						cl.v2BnBeta[c] = beta2 * cl.v2BnBeta[c] + (1.0f - beta2) * g * g;
						const float mH = cl.vBnBeta[c] / bc1;
						const float vH = cl.v2BnBeta[c] / bc2;
						cl.bnBeta[c] -= baseLR * mH / (sqrtf(vH) + epsAdam);
					}
				}
			}

			// Zero gradients for next batch.
			std::memset(&cl.gW[0], 0, cl.gW.size() * sizeof(float));
			std::memset(&cl.gBias[0], 0, cl.gBias.size() * sizeof(float));
			if (!cl.gBnGamma.empty())
				std::memset(&cl.gBnGamma[0], 0, cl.gBnGamma.size() * sizeof(float));
			if (!cl.gBnBeta.empty())
				std::memset(&cl.gBnBeta[0], 0, cl.gBnBeta.size() * sizeof(float));
		}

		// Update FC layers
		for (unsigned int t = 0; t < numFC; ++t)
		{
			TensorCNNState::FCTransition& fc = tensorCnn.fcLayers[t];

			for (size_t i = 0; i < fc.W.size(); ++i)
			{
				float g = fc.gW[i] * invBatch * gradScale;
				fc.vW[i] = beta1 * fc.vW[i] + (1.0f - beta1) * g;
				fc.v2W[i] = beta2 * fc.v2W[i] + (1.0f - beta2) * g * g;
				const float mHat = fc.vW[i] / bc1;
				const float vHat = fc.v2W[i] / bc2;
				fc.W[i] -= baseLR * (mHat / (sqrtf(vHat) + epsAdam) + wd2 * fc.W[i]);
			}

			for (unsigned int j = 0; j < fc.out; ++j)
			{
				float g = fc.gBias[j] * invBatch * gradScale;
				fc.vBias[j] = beta1 * fc.vBias[j] + (1.0f - beta1) * g;
				fc.v2Bias[j] = beta2 * fc.v2Bias[j] + (1.0f - beta2) * g * g;
				const float mHat = fc.vBias[j] / bc1;
				const float vHat = fc.v2Bias[j] / bc2;
				fc.bias[j] -= baseLR * mHat / (sqrtf(vHat) + epsAdam);
			}

			std::memset(&fc.gW[0], 0, fc.gW.size() * sizeof(float));
			std::memset(&fc.gBias[0], 0, fc.gBias.size() * sizeof(float));
		}
	}
	else
	{
		// SGD with momentum
		for (unsigned int l = 0; l < numConv; ++l)
		{
			TensorCNNState::ConvLayer& cl = tensorCnn.convLayers[l];

			for (size_t i = 0; i < cl.W.size(); ++i)
			{
				float g = (cl.gW[i] * invBatch + wd2 * cl.W[i]) * gradScale;
				cl.vW[i] = mf * cl.vW[i] + baseLR * g;
				cl.W[i] -= cl.vW[i];
			}
			for (unsigned int c = 0; c < cl.outC; ++c)
			{
				float g = cl.gBias[c] * invBatch * gradScale;
				cl.vBias[c] = mf * cl.vBias[c] + baseLR * g;
				cl.bias[c] -= cl.vBias[c];
			}

			if (cs.spatialInfo[l].useBatchNorm)
			{
				for (unsigned int c = 0; c < cl.outC; ++c)
				{
					{
						float g = cl.gBnGamma[c] * invBatch * gradScale;
						cl.vBnGamma[c] = mf * cl.vBnGamma[c] + baseLR * g;
						cl.bnGamma[c] -= cl.vBnGamma[c];
					}
					{
						float g = cl.gBnBeta[c] * invBatch * gradScale;
						cl.vBnBeta[c] = mf * cl.vBnBeta[c] + baseLR * g;
						cl.bnBeta[c] -= cl.vBnBeta[c];
					}
				}
			}

			std::memset(&cl.gW[0], 0, cl.gW.size() * sizeof(float));
			std::memset(&cl.gBias[0], 0, cl.gBias.size() * sizeof(float));
			if (!cl.gBnGamma.empty())
				std::memset(&cl.gBnGamma[0], 0, cl.gBnGamma.size() * sizeof(float));
			if (!cl.gBnBeta.empty())
				std::memset(&cl.gBnBeta[0], 0, cl.gBnBeta.size() * sizeof(float));
		}

		for (unsigned int t = 0; t < numFC; ++t)
		{
			TensorCNNState::FCTransition& fc = tensorCnn.fcLayers[t];

			for (size_t i = 0; i < fc.W.size(); ++i)
			{
				float g = (fc.gW[i] * invBatch + wd2 * fc.W[i]) * gradScale;
				fc.vW[i] = mf * fc.vW[i] + baseLR * g;
				fc.W[i] -= fc.vW[i];
			}
			for (unsigned int j = 0; j < fc.out; ++j)
			{
				float g = fc.gBias[j] * invBatch * gradScale;
				fc.vBias[j] = mf * fc.vBias[j] + baseLR * g;
				fc.bias[j] -= fc.vBias[j];
			}

			std::memset(&fc.gW[0], 0, fc.gW.size() * sizeof(float));
			std::memset(&fc.gBias[0], 0, fc.gBias.size() * sizeof(float));
		}
	}

	tensorCnn.batchCount = 0u;
}
