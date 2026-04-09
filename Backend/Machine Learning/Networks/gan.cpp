// GAN (Generative Adversarial Network) implementation.
//
// Supports DFF and CNN architectures for both generator and discriminator,
// with vanilla GAN (BCE) and WGAN-GP loss functions.
//
// The GAN class owns two NNetwork instances and implements its own
// forward/backward/update that operates directly on their tensor state.

#include "gan.h"
#include "glades_thread_pool.h"
#include "sgd_utils.h"
#include "transformer_kernels.h"

#include "../DataObjects/DataInput.h"
#include "../GMath/gmath.h"
#include "../Structure/nninfo.h"
#include "../Structure/inputlayerinfo.h"
#include "../Structure/hiddenlayerinfo.h"
#include "../Structure/outputlayerinfo.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace glades;

// ============================================================
// Local helpers (im2col, col2im, GEMM) - same as sgd_cnn.cpp
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

// Nearest-neighbor upsample: [C, inH, inW] -> [C, inH*scaleH, inW*scaleW]
static void nn_upsample_cpu(const float* input,
                            unsigned int C, unsigned int inH, unsigned int inW,
                            unsigned int scaleH, unsigned int scaleW,
                            float* output)
{
	const unsigned int outH = inH * scaleH;
	const unsigned int outW = inW * scaleW;
	for (unsigned int c = 0; c < C; ++c)
	{
		const size_t inOff = static_cast<size_t>(c) * inH * inW;
		const size_t outOff = static_cast<size_t>(c) * outH * outW;
		for (unsigned int oh = 0; oh < outH; ++oh)
		{
			const unsigned int ih = oh / scaleH;
			for (unsigned int ow = 0; ow < outW; ++ow)
				output[outOff + oh * outW + ow] = input[inOff + ih * inW + ow / scaleW];
		}
	}
}

// Sum-downsample (gradient of nn upsample): [C, inH, inW] -> [C, inH/scaleH, inW/scaleW]
static void nn_downsample_sum_cpu(const float* input,
                                  unsigned int C, unsigned int inH, unsigned int inW,
                                  unsigned int scaleH, unsigned int scaleW,
                                  float* output)
{
	const unsigned int outH = inH / scaleH;
	const unsigned int outW = inW / scaleW;
	std::memset(output, 0, static_cast<size_t>(C) * outH * outW * sizeof(float));
	for (unsigned int c = 0; c < C; ++c)
	{
		const size_t inOff = static_cast<size_t>(c) * inH * inW;
		const size_t outOff = static_cast<size_t>(c) * outH * outW;
		for (unsigned int ih = 0; ih < inH; ++ih)
		{
			const unsigned int oh = ih / scaleH;
			for (unsigned int iw = 0; iw < inW; ++iw)
				output[outOff + oh * outW + iw / scaleW] += input[inOff + ih * inW + iw];
		}
	}
}

// C[M,N] += A[M,K] * B^T[N,K]
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
			for (unsigned int k = 0; k < K; ++k)
				sum += Arow[k] * Brow[k];
			C[m * N + n] += sum;
		}
	}
}

// C[M,N] += A^T[K,M] * B[K,N]. A stored as [K,M].
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
			for (unsigned int n = 0; n < N; ++n)
				Crow[n] += a * Brow[n];
		}
	}
}

// C[M,N] += A[M,K] * B[K,N]
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
			for (unsigned int n = 0; n < N; ++n)
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

// Scratch buffers for CNN forward pass (must be at namespace scope for C++98 compat)
struct GANConvScratch
{
	std::vector<float> im2col;
	std::vector<float> convOut;
	std::vector<float> actOut;
	std::vector<float> poolOut;
	std::vector<int> poolArgmax;
};

// Glorot uniform weight initialization
static void initGlorot(glades::rng::Engine& eng, std::vector<float>& W,
                       unsigned int fanIn, unsigned int fanOut)
{
	if (fanIn == 0u || fanOut == 0u || W.empty())
		return;
	const double limit = sqrt(6.0 / (static_cast<double>(fanIn) + static_cast<double>(fanOut)));
	for (size_t i = 0; i < W.size(); ++i)
		W[i] = static_cast<float>(glades::rng::uniform_double(eng, -limit, limit));
}

// Softmax over a vector (in-place)
static void softmaxVec(float* logits, unsigned int n)
{
	float maxVal = logits[0];
	for (unsigned int i = 1; i < n; ++i)
		if (logits[i] > maxVal) maxVal = logits[i];
	float sum = 0.0f;
	for (unsigned int i = 0; i < n; ++i)
	{
		logits[i] = expf(logits[i] - maxVal);
		sum += logits[i];
	}
	if (sum > 0.0f)
		for (unsigned int i = 0; i < n; ++i)
			logits[i] /= sum;
}

// Categorical cross-entropy: -sum(targets * log(probs))
static float categoricalCrossEntropy(const float* probs, const float* targets, unsigned int n)
{
	const float eps = 1e-7f;
	float loss = 0.0f;
	for (unsigned int i = 0; i < n; ++i)
	{
		const float p = (probs[i] < eps) ? eps : probs[i];
		loss -= targets[i] * logf(p);
	}
	return loss;
}

// Gradient of categorical CE w.r.t. logits (after softmax): probs - targets
static void categoricalCEGrad(const float* probs, const float* targets, float* grad, unsigned int n)
{
	for (unsigned int i = 0; i < n; ++i)
		grad[i] = probs[i] - targets[i];
}

// Gaussian NLL for single (mu, logvar) pair: 0.5*(logvar + (target-mu)^2 / exp(logvar))
static float gaussianNLL(float mu, float logvar, float target)
{
	const float var = expf(logvar);
	const float diff = target - mu;
	return 0.5f * (logvar + diff * diff / (var + 1e-8f));
}

// Gradients of Gaussian NLL w.r.t. mu and logvar
static void gaussianNLLGrad(float mu, float logvar, float target, float& dMu, float& dLogvar)
{
	const float var = expf(logvar);
	const float invVar = 1.0f / (var + 1e-8f);
	const float diff = target - mu;
	dMu = -diff * invVar;
	dLogvar = 0.5f * (1.0f - diff * diff * invVar);
}

// L1 loss: mean(|pred - target|)
static float l1Loss(const float* pred, const float* target, unsigned int dim)
{
	float sum = 0.0f;
	for (unsigned int i = 0; i < dim; ++i)
	{
		float d = pred[i] - target[i];
		sum += (d >= 0.0f) ? d : -d;
	}
	return sum / static_cast<float>(dim);
}

// L1 loss gradient: sign(pred - target) / dim
static void l1LossGrad(const float* pred, const float* target, unsigned int dim, float* grad)
{
	const float invDim = 1.0f / static_cast<float>(dim);
	for (unsigned int i = 0; i < dim; ++i)
	{
		const float d = pred[i] - target[i];
		grad[i] = ((d > 0.0f) ? 1.0f : ((d < 0.0f) ? -1.0f : 0.0f)) * invDim;
	}
}

// Instance normalization: output = (input - mean) / sqrt(var + eps)
static void instanceNorm(const float* input, unsigned int dim, float eps,
                         float* output, float& mean, float& invStd)
{
	mean = 0.0f;
	for (unsigned int i = 0; i < dim; ++i)
		mean += input[i];
	mean /= static_cast<float>(dim);
	float var = 0.0f;
	for (unsigned int i = 0; i < dim; ++i)
	{
		const float d = input[i] - mean;
		var += d * d;
	}
	var /= static_cast<float>(dim);
	invStd = 1.0f / sqrtf(var + eps);
	for (unsigned int i = 0; i < dim; ++i)
		output[i] = (input[i] - mean) * invStd;
}

// Instance norm backward: compute dInput from dOutput and cached xNorm, invStd
static void instanceNormBackward(const float* dOut, const float* xNorm,
                                 float invStd, unsigned int dim, float* dIn)
{
	const float invN = 1.0f / static_cast<float>(dim);
	float sumDOut = 0.0f;
	float sumDOutXN = 0.0f;
	for (unsigned int i = 0; i < dim; ++i)
	{
		sumDOut += dOut[i];
		sumDOutXN += dOut[i] * xNorm[i];
	}
	for (unsigned int i = 0; i < dim; ++i)
		dIn[i] = invStd * (dOut[i] - invN * sumDOut - invN * xNorm[i] * sumDOutXN);
}

} // anonymous namespace

// ============================================================
// Static helpers
// ============================================================

float GAN::sigmoid(float x)
{
	if (x >= 0.0f)
	{
		const float ex = expf(-x);
		return 1.0f / (1.0f + ex);
	}
	else
	{
		const float ex = expf(x);
		return ex / (1.0f + ex);
	}
}

float GAN::sigmoidDeriv(float sigx)
{
	return sigx * (1.0f - sigx);
}

// ============================================================
// GradientBuffer
// ============================================================

void GradientBuffer::initFromDFF(const NNetwork& net)
{
	const NNetwork::TensorDFFState& dff = net.tensorDff;
	const size_t nT = dff.T.size();
	dffGW.resize(nT);
	dffGBias.resize(nT);
	for (size_t t = 0; t < nT; ++t)
	{
		dffGW[t].assign(dff.T[t].gW.size(), 0.0f);
		dffGBias[t].assign(dff.T[t].gBias.size(), 0.0f);
	}
}

void GradientBuffer::initFromCNN(const NNetwork& net)
{
	const NNetwork::TensorCNNState& cnn = net.tensorCnn;
	const size_t nConv = cnn.convLayers.size();
	convGW.resize(nConv);
	convGBias.resize(nConv);
	for (size_t l = 0; l < nConv; ++l)
	{
		convGW[l].assign(cnn.convLayers[l].gW.size(), 0.0f);
		convGBias[l].assign(cnn.convLayers[l].gBias.size(), 0.0f);
	}
	const size_t nFC = cnn.fcLayers.size();
	fcGW.resize(nFC);
	fcGBias.resize(nFC);
	for (size_t l = 0; l < nFC; ++l)
	{
		fcGW[l].assign(cnn.fcLayers[l].gW.size(), 0.0f);
		fcGBias[l].assign(cnn.fcLayers[l].gBias.size(), 0.0f);
	}
}

void GradientBuffer::initFromDeconv(const NNetwork& net)
{
	const NNetwork::TensorDeconvState& dc = net.tensorDeconv;
	deconvFcGW.assign(dc.fcGW.size(), 0.0f);
	deconvFcGBias.assign(dc.fcGBias.size(), 0.0f);
	const size_t nL = dc.layers.size();
	deconvGW.resize(nL);
	deconvGBias.resize(nL);
	deconvGBnGamma.resize(nL);
	deconvGBnBeta.resize(nL);
	for (size_t l = 0; l < nL; ++l)
	{
		deconvGW[l].assign(dc.layers[l].gW.size(), 0.0f);
		deconvGBias[l].assign(dc.layers[l].gBias.size(), 0.0f);
		if (dc.layers[l].useBatchNorm)
		{
			deconvGBnGamma[l].assign(dc.layers[l].bnGamma.size(), 0.0f);
			deconvGBnBeta[l].assign(dc.layers[l].bnBeta.size(), 0.0f);
		}
	}
}

void GradientBuffer::initFromQHead(unsigned int sharedDim, unsigned int qOutDim)
{
	qGW.assign(static_cast<size_t>(sharedDim) * static_cast<size_t>(qOutDim), 0.0f);
	qGBias.assign(qOutDim, 0.0f);
}

void GradientBuffer::initFromGenQHead(unsigned int sharedDim, unsigned int qOutDim, unsigned int hiddenDim)
{
	if (hiddenDim > 0u)
	{
		genQGHiddenW.assign(static_cast<size_t>(hiddenDim) * static_cast<size_t>(sharedDim), 0.0f);
		genQGHiddenBias.assign(hiddenDim, 0.0f);
		genQGW.assign(static_cast<size_t>(qOutDim) * static_cast<size_t>(hiddenDim), 0.0f);
	}
	else
	{
		genQGW.assign(static_cast<size_t>(sharedDim) * static_cast<size_t>(qOutDim), 0.0f);
	}
	genQGBias.assign(qOutDim, 0.0f);
}

void GradientBuffer::zero()
{
	for (size_t t = 0; t < dffGW.size(); ++t)
	{
		if (!dffGW[t].empty())
			std::memset(&dffGW[t][0], 0, dffGW[t].size() * sizeof(float));
		if (!dffGBias[t].empty())
			std::memset(&dffGBias[t][0], 0, dffGBias[t].size() * sizeof(float));
	}
	for (size_t l = 0; l < convGW.size(); ++l)
	{
		if (!convGW[l].empty())
			std::memset(&convGW[l][0], 0, convGW[l].size() * sizeof(float));
		if (!convGBias[l].empty())
			std::memset(&convGBias[l][0], 0, convGBias[l].size() * sizeof(float));
	}
	for (size_t l = 0; l < fcGW.size(); ++l)
	{
		if (!fcGW[l].empty())
			std::memset(&fcGW[l][0], 0, fcGW[l].size() * sizeof(float));
		if (!fcGBias[l].empty())
			std::memset(&fcGBias[l][0], 0, fcGBias[l].size() * sizeof(float));
	}
	if (!deconvFcGW.empty())
		std::memset(&deconvFcGW[0], 0, deconvFcGW.size() * sizeof(float));
	if (!deconvFcGBias.empty())
		std::memset(&deconvFcGBias[0], 0, deconvFcGBias.size() * sizeof(float));
	for (size_t l = 0; l < deconvGW.size(); ++l)
	{
		if (!deconvGW[l].empty())
			std::memset(&deconvGW[l][0], 0, deconvGW[l].size() * sizeof(float));
		if (!deconvGBias[l].empty())
			std::memset(&deconvGBias[l][0], 0, deconvGBias[l].size() * sizeof(float));
	}
	for (size_t l = 0; l < deconvGBnGamma.size(); ++l)
	{
		if (!deconvGBnGamma[l].empty())
			std::memset(&deconvGBnGamma[l][0], 0, deconvGBnGamma[l].size() * sizeof(float));
		if (!deconvGBnBeta[l].empty())
			std::memset(&deconvGBnBeta[l][0], 0, deconvGBnBeta[l].size() * sizeof(float));
	}
	if (!qGW.empty())
		std::memset(&qGW[0], 0, qGW.size() * sizeof(float));
	if (!qGBias.empty())
		std::memset(&qGBias[0], 0, qGBias.size() * sizeof(float));
	if (!genQGW.empty())
		std::memset(&genQGW[0], 0, genQGW.size() * sizeof(float));
	if (!genQGBias.empty())
		std::memset(&genQGBias[0], 0, genQGBias.size() * sizeof(float));
	if (!genQGHiddenW.empty())
		std::memset(&genQGHiddenW[0], 0, genQGHiddenW.size() * sizeof(float));
	if (!genQGHiddenBias.empty())
		std::memset(&genQGHiddenBias[0], 0, genQGHiddenBias.size() * sizeof(float));
}

void GradientBuffer::addToDFF(NNetwork& net) const
{
	NNetwork::TensorDFFState& dff = net.tensorDff;
	for (size_t t = 0; t < dffGW.size(); ++t)
	{
		for (size_t i = 0; i < dffGW[t].size(); ++i)
			dff.T[t].gW[i] += dffGW[t][i];
		for (size_t i = 0; i < dffGBias[t].size(); ++i)
			dff.T[t].gBias[i] += dffGBias[t][i];
	}
}

void GradientBuffer::addToCNN(NNetwork& net) const
{
	NNetwork::TensorCNNState& cnn = net.tensorCnn;
	for (size_t l = 0; l < convGW.size(); ++l)
	{
		for (size_t i = 0; i < convGW[l].size(); ++i)
			cnn.convLayers[l].gW[i] += convGW[l][i];
		for (size_t i = 0; i < convGBias[l].size(); ++i)
			cnn.convLayers[l].gBias[i] += convGBias[l][i];
	}
	for (size_t l = 0; l < fcGW.size(); ++l)
	{
		for (size_t i = 0; i < fcGW[l].size(); ++i)
			cnn.fcLayers[l].gW[i] += fcGW[l][i];
		for (size_t i = 0; i < fcGBias[l].size(); ++i)
			cnn.fcLayers[l].gBias[i] += fcGBias[l][i];
	}
}

void GradientBuffer::addToDeconv(NNetwork& net) const
{
	NNetwork::TensorDeconvState& dc = net.tensorDeconv;
	for (size_t i = 0; i < deconvFcGW.size(); ++i)
		dc.fcGW[i] += deconvFcGW[i];
	for (size_t i = 0; i < deconvFcGBias.size(); ++i)
		dc.fcGBias[i] += deconvFcGBias[i];
	for (size_t l = 0; l < deconvGW.size(); ++l)
	{
		for (size_t i = 0; i < deconvGW[l].size(); ++i)
			dc.layers[l].gW[i] += deconvGW[l][i];
		for (size_t i = 0; i < deconvGBias[l].size(); ++i)
			dc.layers[l].gBias[i] += deconvGBias[l][i];
	}
	for (size_t l = 0; l < deconvGBnGamma.size(); ++l)
	{
		for (size_t i = 0; i < deconvGBnGamma[l].size(); ++i)
			dc.layers[l].gBnGamma[i] += deconvGBnGamma[l][i];
		for (size_t i = 0; i < deconvGBnBeta[l].size(); ++i)
			dc.layers[l].gBnBeta[i] += deconvGBnBeta[l][i];
	}
}

// ============================================================
// GANThreadCtx
// ============================================================

void GANThreadCtx::zeroLosses()
{
	dLossReal = dLossFake = gLoss = wasserstein = infoLoss = divLoss = 0.0f;
	gLossAB = gLossBA = cycleLoss = identityLoss = 0.0f;
	catCorrect = catTotal = 0u;
	dOutRealSum = dOutFakeSum = 0.0f;
}

void DeconvScratchArena::initFromDeconv(const NNetwork& net)
{
	const NNetwork::TensorDeconvState& ds = net.tensorDeconv;
	const unsigned int numLayers = static_cast<unsigned int>(ds.layers.size());

	upsampled.resize(numLayers);
	cols.resize(numLayers);
	outputCols.resize(numLayers);
	dCols.resize(numLayers);
	dUp.resize(numLayers);
	dInput.resize(numLayers);

	for (unsigned int l = 0; l < numLayers; ++l)
	{
		const NNetwork::TensorDeconvState::DeconvLayer& dl = ds.layers[l];

		if (dl.useUpsampleConv)
		{
			const unsigned int upH = dl.inH * dl.strideH;
			const unsigned int upW = dl.inW * dl.strideW;
			const unsigned int K = dl.inC * dl.kH * dl.kW;
			const unsigned int N = dl.outH * dl.outW;

			upsampled[l].resize(static_cast<size_t>(dl.inC) * upH * upW);
			cols[l].resize(static_cast<size_t>(N) * K);
			dCols[l].resize(static_cast<size_t>(N) * K);
			dUp[l].resize(static_cast<size_t>(dl.inC) * upH * upW);
		}
		else
		{
			const unsigned int N = dl.inH * dl.inW;
			const unsigned int outK = dl.outC * dl.kH * dl.kW;

			outputCols[l].resize(static_cast<size_t>(outK) * N);
			dCols[l].resize(static_cast<size_t>(outK) * N);
		}
		dInput[l].resize(static_cast<size_t>(dl.inC) * dl.inH * dl.inW);
	}

	dFC.resize(ds.fcOut);

	unsigned int maxOutVol = 0;
	for (unsigned int l = 0; l < numLayers; ++l)
	{
		const NNetwork::TensorDeconvState::DeconvLayer& dl = ds.layers[l];
		const unsigned int v = dl.outC * dl.outH * dl.outW;
		if (v > maxOutVol) maxOutVol = v;
	}
	dCur.resize(maxOutVol);

	initialized = true;
}

void GANThreadCtx::zeroGrads()
{
	genGrads.zero();
	discGrads.zero();
	qGrads.zero();
	mappingGrads.zero();
	discardDiscGrads.zero();
	for (size_t i = 0; i < styleGW.size(); ++i)
	{
		if (!styleGW[i].empty()) std::memset(&styleGW[i][0], 0, styleGW[i].size() * sizeof(float));
		if (!styleGBias[i].empty()) std::memset(&styleGBias[i][0], 0, styleGBias[i].size() * sizeof(float));
	}
	if (!gNoiseScalesLocal.empty()) std::memset(&gNoiseScalesLocal[0], 0, gNoiseScalesLocal.size() * sizeof(float));
	for (size_t i = 0; i < lnGGamma.size(); ++i)
	{
		if (!lnGGamma[i].empty()) std::memset(&lnGGamma[i][0], 0, lnGGamma[i].size() * sizeof(float));
		if (!lnGBeta[i].empty()) std::memset(&lnGBeta[i][0], 0, lnGBeta[i].size() * sizeof(float));
	}
	// CycleGAN-specific buffers
	genBAGrads.zero();
	discBGrads.zero();
	qBGrads.zero();
	mappingBAGrads.zero();
	discardDiscBGrads.zero();
	for (size_t i = 0; i < styleBAGW.size(); ++i)
	{
		if (!styleBAGW[i].empty()) std::memset(&styleBAGW[i][0], 0, styleBAGW[i].size() * sizeof(float));
		if (!styleBAGBias[i].empty()) std::memset(&styleBAGBias[i][0], 0, styleBAGBias[i].size() * sizeof(float));
	}
	if (!gNoiseScalesBALocal.empty()) std::memset(&gNoiseScalesBALocal[0], 0, gNoiseScalesBALocal.size() * sizeof(float));
	for (size_t i = 0; i < lnBAGGamma.size(); ++i)
	{
		if (!lnBAGGamma[i].empty()) std::memset(&lnBAGGamma[i][0], 0, lnBAGGamma[i].size() * sizeof(float));
		if (!lnBAGBeta[i].empty()) std::memset(&lnBAGBeta[i][0], 0, lnBAGBeta[i].size() * sizeof(float));
	}
}

// ============================================================
// Constructor / Destructor
// ============================================================

GAN::GAN(const GANConfig& cfg,
         const NNInfo* generatorInfo,
         const NNInfo* discriminatorInfo)
	: config(cfg),
	  generator(generatorInfo, NNetwork::TYPE_DFF),
	  discriminator(discriminatorInfo, NNetwork::TYPE_DFF),
	  mappingNet(NNetwork::TYPE_DFF),
	  generatorBA(NNetwork::TYPE_DFF),
	  discriminatorB(NNetwork::TYPE_DFF),
	  mappingNetBA(NNetwork::TYPE_DFF)
{
	glades::rng::seed_engine(rngEngine, 42ULL);
}

GAN::GAN(const GANConfig& cfg,
         const NNInfo* generatorInfo,
         const NNInfo* discriminatorInfo,
         const NNInfo* mappingInfo)
	: config(cfg),
	  generator(generatorInfo, NNetwork::TYPE_DFF),
	  discriminator(discriminatorInfo, NNetwork::TYPE_DFF),
	  mappingNet(mappingInfo, NNetwork::TYPE_DFF),
	  generatorBA(NNetwork::TYPE_DFF),
	  discriminatorB(NNetwork::TYPE_DFF),
	  mappingNetBA(NNetwork::TYPE_DFF)
{
	glades::rng::seed_engine(rngEngine, 42ULL);
}

GAN::GAN(const GANConfig& cfg,
         const NNInfo* genABInfo,
         const NNInfo* genBAInfo,
         const NNInfo* discAInfo,
         const NNInfo* discBInfo)
	: config(cfg),
	  generator(genABInfo, NNetwork::TYPE_DFF),
	  discriminator(discAInfo, NNetwork::TYPE_DFF),
	  mappingNet(NNetwork::TYPE_DFF),
	  generatorBA(genBAInfo, NNetwork::TYPE_DFF),
	  discriminatorB(discBInfo, NNetwork::TYPE_DFF),
	  mappingNetBA(NNetwork::TYPE_DFF)
{
	glades::rng::seed_engine(rngEngine, 42ULL);
}

GAN::GAN(const GANConfig& cfg,
         const NNInfo* genABInfo,
         const NNInfo* genBAInfo,
         const NNInfo* discAInfo,
         const NNInfo* discBInfo,
         const NNInfo* mapABInfo,
         const NNInfo* mapBAInfo)
	: config(cfg),
	  generator(genABInfo, NNetwork::TYPE_DFF),
	  discriminator(discAInfo, NNetwork::TYPE_DFF),
	  mappingNet(mapABInfo, NNetwork::TYPE_DFF),
	  generatorBA(genBAInfo, NNetwork::TYPE_DFF),
	  discriminatorB(discBInfo, NNetwork::TYPE_DFF),
	  mappingNetBA(mapBAInfo, NNetwork::TYPE_DFF)
{
	glades::rng::seed_engine(rngEngine, 42ULL);
}

GAN::~GAN()
{
}

void GAN::setSeed(uint64_t seed)
{
	const bool hasStyle = config.useStyle || (config.variantType == GANConfig::GAN_STYLE);
	const bool hasCycle = config.useCycle || (config.variantType == GANConfig::GAN_CYCLE);

	glades::rng::seed_engine(rngEngine, seed);
	generator.setSeed(seed);
	discriminator.setSeed(seed + 1ULL);
	if (hasStyle)
		mappingNet.setSeed(seed + 2ULL);
	if (hasCycle)
	{
		generatorBA.setSeed(seed + 2ULL);
		discriminatorB.setSeed(seed + 3ULL);
		if (hasStyle)
			mappingNetBA.setSeed(seed + 4ULL);
	}
}

const NNetwork& GAN::getGenerator() const
{
	return generator;
}

const NNetwork& GAN::getDiscriminator() const
{
	return discriminator;
}

const NNetwork& GAN::getGeneratorBA() const
{
	return generatorBA;
}

const NNetwork& GAN::getDiscriminatorB() const
{
	return discriminatorB;
}

// ============================================================
// Weight serialization
// ============================================================

int GAN::saveWeights(const std::string& dir, const std::string& prefix) const
{
	int saved = 0;
	const char* names[] = { "genAB", "genBA", "discA", "discB" };
	const NNetwork* nets[] = { &generator, &generatorBA, &discriminator, &discriminatorB };
	const bool isCycle = (config.variantType == GANConfig::GAN_CYCLE);

	for (int n = 0; n < 4; ++n)
	{
		if (!isCycle && (n == 1 || n == 3)) continue;
		std::string path = dir + "/" + prefix + "_" + names[n] + ".bin";
		FILE* fp = fopen(path.c_str(), "wb");
		if (!fp) continue;

		const NNetwork& net = *nets[n];

		// DFF weights
		uint32_t numT = static_cast<uint32_t>(net.tensorDff.T.size());
		fwrite(&numT, sizeof(uint32_t), 1, fp);
		for (uint32_t t = 0; t < numT; ++t)
		{
			uint32_t dims[2] = { net.tensorDff.T[t].in, net.tensorDff.T[t].out };
			fwrite(dims, sizeof(uint32_t), 2, fp);
			fwrite(net.tensorDff.T[t].W.data(), sizeof(float), net.tensorDff.T[t].W.size(), fp);
			fwrite(net.tensorDff.T[t].bias.data(), sizeof(float), net.tensorDff.T[t].bias.size(), fp);
		}

		// CNN conv layers
		uint32_t numConv = static_cast<uint32_t>(net.tensorCnn.convLayers.size());
		fwrite(&numConv, sizeof(uint32_t), 1, fp);
		for (uint32_t l = 0; l < numConv; ++l)
		{
			uint32_t cdims[4] = { net.tensorCnn.convLayers[l].inC, net.tensorCnn.convLayers[l].outC,
			                      net.tensorCnn.convLayers[l].kH, net.tensorCnn.convLayers[l].kW };
			fwrite(cdims, sizeof(uint32_t), 4, fp);
			fwrite(net.tensorCnn.convLayers[l].W.data(), sizeof(float), net.tensorCnn.convLayers[l].W.size(), fp);
			fwrite(net.tensorCnn.convLayers[l].bias.data(), sizeof(float), net.tensorCnn.convLayers[l].bias.size(), fp);
		}

		// CNN FC layers
		uint32_t numFC = static_cast<uint32_t>(net.tensorCnn.fcLayers.size());
		fwrite(&numFC, sizeof(uint32_t), 1, fp);
		for (uint32_t t = 0; t < numFC; ++t)
		{
			uint32_t dims[2] = { net.tensorCnn.fcLayers[t].in, net.tensorCnn.fcLayers[t].out };
			fwrite(dims, sizeof(uint32_t), 2, fp);
			fwrite(net.tensorCnn.fcLayers[t].W.data(), sizeof(float), net.tensorCnn.fcLayers[t].W.size(), fp);
			fwrite(net.tensorCnn.fcLayers[t].bias.data(), sizeof(float), net.tensorCnn.fcLayers[t].bias.size(), fp);
		}

		// Deconv state (generator)
		uint32_t deconvFC[2] = { net.tensorDeconv.fcIn, net.tensorDeconv.fcOut };
		fwrite(deconvFC, sizeof(uint32_t), 2, fp);
		if (net.tensorDeconv.fcOut > 0)
		{
			fwrite(net.tensorDeconv.fcW.data(), sizeof(float), net.tensorDeconv.fcW.size(), fp);
			fwrite(net.tensorDeconv.fcBias.data(), sizeof(float), net.tensorDeconv.fcBias.size(), fp);
			uint32_t proj[3] = { net.tensorDeconv.projectC, net.tensorDeconv.projectH, net.tensorDeconv.projectW };
			fwrite(proj, sizeof(uint32_t), 3, fp);
		}
		uint32_t numDeconv = static_cast<uint32_t>(net.tensorDeconv.layers.size());
		fwrite(&numDeconv, sizeof(uint32_t), 1, fp);
		for (uint32_t l = 0; l < numDeconv; ++l)
		{
			uint32_t ddims[11] = {
				net.tensorDeconv.layers[l].inC, net.tensorDeconv.layers[l].outC,
				net.tensorDeconv.layers[l].kH, net.tensorDeconv.layers[l].kW,
				net.tensorDeconv.layers[l].strideH, net.tensorDeconv.layers[l].strideW,
				net.tensorDeconv.layers[l].padH, net.tensorDeconv.layers[l].padW,
				net.tensorDeconv.layers[l].useUpsampleConv ? 1u : 0u,
				net.tensorDeconv.layers[l].useReLU ? 1u : 0u,
				net.tensorDeconv.layers[l].useTanh ? 1u : 0u
			};
			fwrite(ddims, sizeof(uint32_t), 11, fp);
			fwrite(net.tensorDeconv.layers[l].W.data(), sizeof(float), net.tensorDeconv.layers[l].W.size(), fp);
			fwrite(net.tensorDeconv.layers[l].bias.data(), sizeof(float), net.tensorDeconv.layers[l].bias.size(), fp);
			uint8_t hasBN = net.tensorDeconv.layers[l].useBatchNorm ? 1u : 0u;
			fwrite(&hasBN, sizeof(uint8_t), 1, fp);
			if (hasBN)
			{
				fwrite(net.tensorDeconv.layers[l].bnGamma.data(), sizeof(float), net.tensorDeconv.layers[l].bnGamma.size(), fp);
				fwrite(net.tensorDeconv.layers[l].bnBeta.data(), sizeof(float), net.tensorDeconv.layers[l].bnBeta.size(), fp);
			}
		}

		fclose(fp);
		++saved;
	}
	return saved;
}

int GAN::loadWeights(const std::string& dir, const std::string& prefix)
{
	int loaded = 0;
	const char* names[] = { "genAB", "genBA", "discA", "discB" };
	NNetwork* nets[] = { &generator, &generatorBA, &discriminator, &discriminatorB };
	const bool isCycle = (config.variantType == GANConfig::GAN_CYCLE);

	for (int n = 0; n < 4; ++n)
	{
		if (!isCycle && (n == 1 || n == 3)) continue;
		std::string path = dir + "/" + prefix + "_" + names[n] + ".bin";
		FILE* fp = fopen(path.c_str(), "rb");
		if (!fp) continue;

		NNetwork& net = *nets[n];

		// DFF weights
		uint32_t numT = 0;
		if (fread(&numT, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); continue; }
		net.tensorDff.T.resize(numT);
		for (uint32_t t = 0; t < numT; ++t)
		{
			uint32_t dims[2];
			if (fread(dims, sizeof(uint32_t), 2, fp) != 2) break;
			net.tensorDff.T[t].in = dims[0];
			net.tensorDff.T[t].out = dims[1];
			net.tensorDff.T[t].W.resize(static_cast<size_t>(dims[0]) * dims[1]);
			net.tensorDff.T[t].bias.resize(dims[1]);
			net.tensorDff.T[t].gW.resize(net.tensorDff.T[t].W.size(), 0.0f);
			net.tensorDff.T[t].gBias.resize(dims[1], 0.0f);
			fread(net.tensorDff.T[t].W.data(), sizeof(float), net.tensorDff.T[t].W.size(), fp);
			fread(net.tensorDff.T[t].bias.data(), sizeof(float), net.tensorDff.T[t].bias.size(), fp);
		}
		if (numT > 0) net.tensorDff.initialized = true;

		// CNN conv layers
		uint32_t numConv = 0;
		if (fread(&numConv, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); continue; }
		net.tensorCnn.convLayers.resize(numConv);
		for (uint32_t l = 0; l < numConv; ++l)
		{
			uint32_t cdims[4];
			if (fread(cdims, sizeof(uint32_t), 4, fp) != 4) break;
			NNetwork::TensorCNNState::ConvLayer& cl = net.tensorCnn.convLayers[l];
			cl.inC = cdims[0]; cl.outC = cdims[1]; cl.kH = cdims[2]; cl.kW = cdims[3];
			cl.W.resize(static_cast<size_t>(cl.outC) * cl.inC * cl.kH * cl.kW);
			cl.bias.resize(cl.outC);
			cl.gW.resize(cl.W.size(), 0.0f);
			cl.gBias.resize(cl.outC, 0.0f);
			fread(cl.W.data(), sizeof(float), cl.W.size(), fp);
			fread(cl.bias.data(), sizeof(float), cl.bias.size(), fp);
		}

		// CNN FC layers
		uint32_t numFC = 0;
		if (fread(&numFC, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); continue; }
		net.tensorCnn.fcLayers.resize(numFC);
		for (uint32_t t = 0; t < numFC; ++t)
		{
			uint32_t dims[2];
			if (fread(dims, sizeof(uint32_t), 2, fp) != 2) break;
			NNetwork::TensorCNNState::FCTransition& fc = net.tensorCnn.fcLayers[t];
			fc.in = dims[0]; fc.out = dims[1];
			fc.W.resize(static_cast<size_t>(dims[0]) * dims[1]);
			fc.bias.resize(dims[1]);
			fc.gW.resize(fc.W.size(), 0.0f);
			fc.gBias.resize(dims[1], 0.0f);
			fread(fc.W.data(), sizeof(float), fc.W.size(), fp);
			fread(fc.bias.data(), sizeof(float), fc.bias.size(), fp);
		}
		if (numConv > 0 || numFC > 0) net.tensorCnn.initialized = true;

		// Deconv state
		uint32_t deconvFC[2] = { 0, 0 };
		if (fread(deconvFC, sizeof(uint32_t), 2, fp) != 2) { fclose(fp); continue; }
		net.tensorDeconv.fcIn = deconvFC[0];
		net.tensorDeconv.fcOut = deconvFC[1];
		if (net.tensorDeconv.fcOut > 0)
		{
			net.tensorDeconv.fcW.resize(static_cast<size_t>(deconvFC[0]) * deconvFC[1]);
			net.tensorDeconv.fcBias.resize(deconvFC[1]);
			net.tensorDeconv.fcGW.resize(net.tensorDeconv.fcW.size(), 0.0f);
			net.tensorDeconv.fcGBias.resize(deconvFC[1], 0.0f);
			fread(net.tensorDeconv.fcW.data(), sizeof(float), net.tensorDeconv.fcW.size(), fp);
			fread(net.tensorDeconv.fcBias.data(), sizeof(float), net.tensorDeconv.fcBias.size(), fp);
			uint32_t proj[3];
			if (fread(proj, sizeof(uint32_t), 3, fp) == 3)
			{
				net.tensorDeconv.projectC = proj[0];
				net.tensorDeconv.projectH = proj[1];
				net.tensorDeconv.projectW = proj[2];
			}
		}
		uint32_t numDeconv = 0;
		if (fread(&numDeconv, sizeof(uint32_t), 1, fp) != 1) { fclose(fp); continue; }
		net.tensorDeconv.layers.resize(numDeconv);
		for (uint32_t l = 0; l < numDeconv; ++l)
		{
			uint32_t ddims[11];
			if (fread(ddims, sizeof(uint32_t), 11, fp) != 11) break;
			NNetwork::TensorDeconvState::DeconvLayer& dl = net.tensorDeconv.layers[l];
			dl.inC = ddims[0]; dl.outC = ddims[1];
			dl.kH = ddims[2]; dl.kW = ddims[3];
			dl.strideH = ddims[4]; dl.strideW = ddims[5];
			dl.padH = ddims[6]; dl.padW = ddims[7];
			dl.useUpsampleConv = (ddims[8] != 0u);
			dl.useReLU = (ddims[9] != 0u);
			dl.useTanh = (ddims[10] != 0u);
			dl.W.resize(static_cast<size_t>(dl.inC) * dl.outC * dl.kH * dl.kW);
			dl.bias.resize(dl.outC);
			dl.gW.resize(dl.W.size(), 0.0f);
			dl.gBias.resize(dl.outC, 0.0f);
			fread(dl.W.data(), sizeof(float), dl.W.size(), fp);
			fread(dl.bias.data(), sizeof(float), dl.bias.size(), fp);
			uint8_t hasBN = 0;
			if (fread(&hasBN, sizeof(uint8_t), 1, fp) == 1 && hasBN)
			{
				dl.useBatchNorm = true;
				dl.bnGamma.resize(dl.outC);
				dl.bnBeta.resize(dl.outC);
				dl.gBnGamma.resize(dl.outC, 0.0f);
				dl.gBnBeta.resize(dl.outC, 0.0f);
				dl.vBnGamma.resize(dl.outC, 0.0f);
				dl.v2BnGamma.resize(dl.outC, 0.0f);
				dl.vBnBeta.resize(dl.outC, 0.0f);
				dl.v2BnBeta.resize(dl.outC, 0.0f);
				fread(dl.bnGamma.data(), sizeof(float), dl.bnGamma.size(), fp);
				fread(dl.bnBeta.data(), sizeof(float), dl.bnBeta.size(), fp);
			}
		}
		if (net.tensorDeconv.fcOut > 0 || numDeconv > 0)
		{
			// Recompute spatial dimensions (not stored in binary)
			unsigned int curH = net.tensorDeconv.projectH;
			unsigned int curW = net.tensorDeconv.projectW;
			for (uint32_t l = 0; l < numDeconv; ++l)
			{
				NNetwork::TensorDeconvState::DeconvLayer& dl = net.tensorDeconv.layers[l];
				dl.inH = curH;
				dl.inW = curW;
				if (dl.useUpsampleConv)
				{
					const unsigned int upH = curH * dl.strideH;
					const unsigned int upW = curW * dl.strideW;
					dl.outH = upH - dl.kH + 2u * dl.padH + 1u;
					dl.outW = upW - dl.kW + 2u * dl.padW + 1u;
				}
				else
				{
					dl.outH = (curH - 1u) * dl.strideH - 2u * dl.padH + dl.kH;
					dl.outW = (curW - 1u) * dl.strideW - 2u * dl.padW + dl.kW;
				}
				curH = dl.outH;
				curW = dl.outW;
			}
			net.tensorDeconv.initialized = true;
		}

		fclose(fp);
		++loaded;
	}
	return loaded;
}

// ============================================================
// Gradient zeroing utilities
// ============================================================

void GAN::zeroDFFGrads(NNetwork& net)
{
	for (unsigned int t = 0; t < net.tensorDff.T.size(); ++t)
	{
		NNetwork::TensorDFFState::Transition& tr = net.tensorDff.T[t];
		std::memset(&tr.gW[0], 0, tr.gW.size() * sizeof(float));
		std::memset(&tr.gBias[0], 0, tr.gBias.size() * sizeof(float));
	}
}

void GAN::zeroCNNGrads(NNetwork& net)
{
	for (unsigned int l = 0; l < net.tensorCnn.convLayers.size(); ++l)
	{
		NNetwork::TensorCNNState::ConvLayer& cl = net.tensorCnn.convLayers[l];
		std::memset(&cl.gW[0], 0, cl.gW.size() * sizeof(float));
		std::memset(&cl.gBias[0], 0, cl.gBias.size() * sizeof(float));
	}
	for (unsigned int t = 0; t < net.tensorCnn.fcLayers.size(); ++t)
	{
		NNetwork::TensorCNNState::FCTransition& fc = net.tensorCnn.fcLayers[t];
		std::memset(&fc.gW[0], 0, fc.gW.size() * sizeof(float));
		std::memset(&fc.gBias[0], 0, fc.gBias.size() * sizeof(float));
	}
}

void GAN::zeroDiscGrads()
{
	if (config.archType == GANConfig::GAN_DFF)
		zeroDFFGrads(discriminator);
	else
		zeroCNNGrads(discriminator);
}

// ============================================================
// Spectral normalization (power iteration)
// ============================================================

void GAN::spectralNormDFF(NNetwork& net, SpectralNormState& state, int nIters)
{
	const unsigned int numT = static_cast<unsigned int>(net.tensorDff.T.size());
	if (!state.initialized)
	{
		state.snU.resize(numT);
		for (unsigned int t = 0; t < numT; ++t)
		{
			const unsigned int outDim = net.tensorDff.T[t].out;
			state.snU[t].resize(outDim);
			// Initialize u as unit vector: u = (1/sqrt(n), ..., 1/sqrt(n))
			const float initVal = 1.0f / std::sqrt(static_cast<float>(outDim));
			for (unsigned int i = 0; i < outDim; ++i)
				state.snU[t][i] = initVal;
		}
		state.initialized = true;
	}

	for (unsigned int t = 0; t < numT; ++t)
	{
		NNetwork::TensorDFFState::Transition& tr = net.tensorDff.T[t];
		const unsigned int outDim = tr.out;
		const unsigned int inDim = tr.in;
		float* W = &tr.W[0];
		float* u = &state.snU[t][0];

		std::vector<float> v(inDim, 0.0f);

		for (int iter = 0; iter < nIters; ++iter)
		{
			// v = W^T @ u
			for (unsigned int j = 0; j < inDim; ++j)
			{
				float sum = 0.0f;
				for (unsigned int i = 0; i < outDim; ++i)
					sum += W[i * inDim + j] * u[i];
				v[j] = sum;
			}
			// Normalize v
			float normV = 0.0f;
			for (unsigned int j = 0; j < inDim; ++j)
				normV += v[j] * v[j];
			normV = std::sqrt(normV) + 1e-12f;
			for (unsigned int j = 0; j < inDim; ++j)
				v[j] /= normV;

			// u = W @ v
			for (unsigned int i = 0; i < outDim; ++i)
			{
				float sum = 0.0f;
				for (unsigned int j = 0; j < inDim; ++j)
					sum += W[i * inDim + j] * v[j];
				u[i] = sum;
			}
			// Normalize u
			float normU = 0.0f;
			for (unsigned int i = 0; i < outDim; ++i)
				normU += u[i] * u[i];
			normU = std::sqrt(normU) + 1e-12f;
			for (unsigned int i = 0; i < outDim; ++i)
				u[i] /= normU;
		}

		// sigma = u^T W v
		float sigma = 0.0f;
		for (unsigned int i = 0; i < outDim; ++i)
		{
			float Wv_i = 0.0f;
			for (unsigned int j = 0; j < inDim; ++j)
				Wv_i += W[i * inDim + j] * v[j];
			sigma += u[i] * Wv_i;
		}

		// Divide all weights by sigma
		if (sigma > 1e-12f)
		{
			const float invSigma = 1.0f / sigma;
			const unsigned int wSize = outDim * inDim;
			for (unsigned int k = 0; k < wSize; ++k)
				W[k] *= invSigma;
		}
	}
}

void GAN::spectralNormCNN(NNetwork& net, SpectralNormState& state, int nIters)
{
	const unsigned int numConv = static_cast<unsigned int>(net.tensorCnn.convLayers.size());
	const unsigned int numFC = static_cast<unsigned int>(net.tensorCnn.fcLayers.size());

	if (!state.initialized)
	{
		state.snConvU.resize(numConv);
		for (unsigned int l = 0; l < numConv; ++l)
		{
			const unsigned int outC = net.tensorCnn.convLayers[l].outC;
			state.snConvU[l].resize(outC);
			const float initVal = 1.0f / std::sqrt(static_cast<float>(outC));
			for (unsigned int i = 0; i < outC; ++i)
				state.snConvU[l][i] = initVal;
		}
		state.snFCU.resize(numFC);
		for (unsigned int t = 0; t < numFC; ++t)
		{
			const unsigned int outDim = net.tensorCnn.fcLayers[t].out;
			state.snFCU[t].resize(outDim);
			const float initVal = 1.0f / std::sqrt(static_cast<float>(outDim));
			for (unsigned int i = 0; i < outDim; ++i)
				state.snFCU[t][i] = initVal;
		}
		state.initialized = true;
	}

	// Spectral norm for conv layers: treat W as [outC, inC*kH*kW]
	for (unsigned int l = 0; l < numConv; ++l)
	{
		NNetwork::TensorCNNState::ConvLayer& cl = net.tensorCnn.convLayers[l];
		const unsigned int outC = cl.outC;
		const unsigned int K = cl.inC * cl.kH * cl.kW; // fan-in per filter
		float* W = &cl.W[0];
		float* u = &state.snConvU[l][0];

		std::vector<float> v(K, 0.0f);

		for (int iter = 0; iter < nIters; ++iter)
		{
			// v = W^T @ u
			for (unsigned int j = 0; j < K; ++j)
			{
				float sum = 0.0f;
				for (unsigned int i = 0; i < outC; ++i)
					sum += W[i * K + j] * u[i];
				v[j] = sum;
			}
			float normV = 0.0f;
			for (unsigned int j = 0; j < K; ++j)
				normV += v[j] * v[j];
			normV = std::sqrt(normV) + 1e-12f;
			for (unsigned int j = 0; j < K; ++j)
				v[j] /= normV;

			// u = W @ v
			for (unsigned int i = 0; i < outC; ++i)
			{
				float sum = 0.0f;
				for (unsigned int j = 0; j < K; ++j)
					sum += W[i * K + j] * v[j];
				u[i] = sum;
			}
			float normU = 0.0f;
			for (unsigned int i = 0; i < outC; ++i)
				normU += u[i] * u[i];
			normU = std::sqrt(normU) + 1e-12f;
			for (unsigned int i = 0; i < outC; ++i)
				u[i] /= normU;
		}

		float sigma = 0.0f;
		for (unsigned int i = 0; i < outC; ++i)
		{
			float Wv_i = 0.0f;
			for (unsigned int j = 0; j < K; ++j)
				Wv_i += W[i * K + j] * v[j];
			sigma += u[i] * Wv_i;
		}

		if (sigma > 1e-12f)
		{
			const float invSigma = 1.0f / sigma;
			const unsigned int wSize = outC * K;
			for (unsigned int k = 0; k < wSize; ++k)
				W[k] *= invSigma;
		}
	}

	// Spectral norm for FC layers: treat W as [out, in]
	for (unsigned int t = 0; t < numFC; ++t)
	{
		NNetwork::TensorCNNState::FCTransition& fc = net.tensorCnn.fcLayers[t];
		const unsigned int outDim = fc.out;
		const unsigned int inDim = fc.in;
		float* W = &fc.W[0];
		float* u = &state.snFCU[t][0];

		std::vector<float> v(inDim, 0.0f);

		for (int iter = 0; iter < nIters; ++iter)
		{
			for (unsigned int j = 0; j < inDim; ++j)
			{
				float sum = 0.0f;
				for (unsigned int i = 0; i < outDim; ++i)
					sum += W[i * inDim + j] * u[i];
				v[j] = sum;
			}
			float normV = 0.0f;
			for (unsigned int j = 0; j < inDim; ++j)
				normV += v[j] * v[j];
			normV = std::sqrt(normV) + 1e-12f;
			for (unsigned int j = 0; j < inDim; ++j)
				v[j] /= normV;

			for (unsigned int i = 0; i < outDim; ++i)
			{
				float sum = 0.0f;
				for (unsigned int j = 0; j < inDim; ++j)
					sum += W[i * inDim + j] * v[j];
				u[i] = sum;
			}
			float normU = 0.0f;
			for (unsigned int i = 0; i < outDim; ++i)
				normU += u[i] * u[i];
			normU = std::sqrt(normU) + 1e-12f;
			for (unsigned int i = 0; i < outDim; ++i)
				u[i] /= normU;
		}

		float sigma = 0.0f;
		for (unsigned int i = 0; i < outDim; ++i)
		{
			float Wv_i = 0.0f;
			for (unsigned int j = 0; j < inDim; ++j)
				Wv_i += W[i * inDim + j] * v[j];
			sigma += u[i] * Wv_i;
		}

		if (sigma > 1e-12f)
		{
			const float invSigma = 1.0f / sigma;
			const unsigned int wSize = outDim * inDim;
			for (unsigned int k = 0; k < wSize; ++k)
				W[k] *= invSigma;
		}
	}
}

// ============================================================
// Noise sampling
// ============================================================

void GAN::sampleNoise(std::vector<float>& noise) const
{
	noise.resize(config.noiseDim);
	// const_cast needed because rng::standard_normal mutates engine state
	glades::rng::Engine& eng = const_cast<glades::rng::Engine&>(rngEngine);
	for (unsigned int i = 0; i < config.noiseDim; ++i)
		noise[i] = glades::rng::standard_normal(eng);
}

// ============================================================
// DFF Tensor initialization
// ============================================================

bool GAN::initDFFTensors(NNetwork& net, unsigned int inputSize, unsigned int outputSize)
{
	NNInfo* skel = net.skeleton;
	if (!skel)
		return false;

	const int H = skel->numHiddenLayers();

	std::vector<unsigned int> wantSizes;
	wantSizes.push_back(inputSize);
	for (int l = 0; l < H; ++l)
	{
		const int hs = skel->getHiddenLayerSize(static_cast<unsigned int>(l));
		wantSizes.push_back(hs > 0 ? static_cast<unsigned int>(hs) : 0u);
	}
	wantSizes.push_back(outputSize);

	for (size_t i = 0; i < wantSizes.size(); ++i)
	{
		if (wantSizes[i] == 0u)
			return false;
	}

	if (net.tensorDff.initialized && net.tensorDff.sizes == wantSizes)
		return true;

	net.tensorDff.reset();
	net.tensorDff.sizes = wantSizes;

	const unsigned int numTransitions =
	    (wantSizes.size() >= 2u) ? static_cast<unsigned int>(wantSizes.size() - 1u) : 0u;
	net.tensorDff.T.resize(numTransitions);
	net.tensorDff.a.resize(wantSizes.size());
	net.tensorDff.delta.resize(wantSizes.size());

	for (unsigned int li = 0; li < wantSizes.size(); ++li)
	{
		net.tensorDff.a[li].assign(wantSizes[li], 0.0f);
		if (li == 0u)
			net.tensorDff.delta[li].clear();
		else
			net.tensorDff.delta[li].assign(wantSizes[li], 0.0f);
	}

	for (unsigned int t = 0; t < numTransitions; ++t)
	{
		const unsigned int in = wantSizes[t];
		const unsigned int out = wantSizes[t + 1u];
		NNetwork::TensorDFFState::Transition& tr = net.tensorDff.T[t];
		tr.in = in;
		tr.out = out;
		tr.W.assign(static_cast<size_t>(out) * static_cast<size_t>(in), 0.0f);
		tr.vW.assign(tr.W.size(), 0.0f);
		tr.gW.assign(tr.W.size(), 0.0f);
		tr.bias.assign(out, 0.0f);
		tr.gBias.assign(out, 0.0f);

		initGlorot(rngEngine, tr.W, in, out);
	}

	net.tensorDff.batchCount = 0u;
	net.tensorDff.initialized = true;
	return true;
}

// ============================================================
// CNN Tensor initialization
// ============================================================

bool GAN::initCNNTensors(NNetwork& net, const CNNConfig& cnnCfg, unsigned int outputSize)
{
	NNInfo* skel = net.skeleton;
	if (!skel)
		return false;

	NNetwork::TensorCNNState& cs = net.tensorCnn;
	if (cs.initialized)
		return true;

	cs.inputH = cnnCfg.inputH;
	cs.inputW = cnnCfg.inputW;
	cs.inputC = cnnCfg.inputC;

	const unsigned int numConv = static_cast<unsigned int>(cnnCfg.convLayers.size());
	cs.convLayers.resize(numConv);
	cs.spatialInfo.resize(numConv);

	unsigned int curH = cs.inputH, curW = cs.inputW, curC = cs.inputC;

	for (unsigned int l = 0; l < numConv; ++l)
	{
		const CNNConfig::ConvLayerSpec& spec = cnnCfg.convLayers[l];
		NNetwork::TensorCNNState::ConvSpatialInfo& sp = cs.spatialInfo[l];
		NNetwork::TensorCNNState::ConvLayer& cl = cs.convLayers[l];

		sp.inH = curH; sp.inW = curW; sp.inC = curC;
		sp.kH = spec.kernelH; sp.kW = spec.kernelW;
		sp.strideH = spec.strideH; sp.strideW = spec.strideW;
		sp.padH = spec.padH; sp.padW = spec.padW;
		sp.outC = spec.outChannels;
		sp.outH = (sp.inH + 2 * sp.padH - sp.kH) / sp.strideH + 1;
		sp.outW = (sp.inW + 2 * sp.padW - sp.kW) / sp.strideW + 1;
		sp.useBatchNorm = spec.useBatchNorm;
		sp.useMaxPool = spec.useMaxPool;
		sp.poolH = spec.poolH; sp.poolW = spec.poolW;
		sp.poolStrideH = spec.poolStrideH; sp.poolStrideW = spec.poolStrideW;

		if (sp.useMaxPool)
		{
			sp.poolOutH = (sp.outH - sp.poolH) / sp.poolStrideH + 1;
			sp.poolOutW = (sp.outW - sp.poolW) / sp.poolStrideW + 1;
		}
		else
		{
			sp.poolOutH = sp.outH;
			sp.poolOutW = sp.outW;
		}

		sp.im2colRows = sp.outH * sp.outW;
		sp.im2colCols = sp.inC * sp.kH * sp.kW;

		cl.outC = sp.outC; cl.inC = sp.inC; cl.kH = sp.kH; cl.kW = sp.kW;
		const size_t wSize = static_cast<size_t>(sp.outC) * sp.im2colCols;
		cl.W.assign(wSize, 0.0f);
		cl.bias.assign(sp.outC, 0.0f);
		cl.gW.assign(wSize, 0.0f);
		cl.gBias.assign(sp.outC, 0.0f);
		cl.vW.assign(wSize, 0.0f);
		cl.v2W.assign(wSize, 0.0f);
		cl.vBias.assign(sp.outC, 0.0f);
		cl.v2Bias.assign(sp.outC, 0.0f);

		if (sp.useBatchNorm)
		{
			cl.bnGamma.assign(sp.outC, 1.0f);
			cl.bnBeta.assign(sp.outC, 0.0f);
			cl.bnRunMean.assign(sp.outC, 0.0f);
			cl.bnRunVar.assign(sp.outC, 1.0f);
			cl.gBnGamma.assign(sp.outC, 0.0f);
			cl.gBnBeta.assign(sp.outC, 0.0f);
			cl.vBnGamma.assign(sp.outC, 0.0f);
			cl.v2BnGamma.assign(sp.outC, 0.0f);
			cl.vBnBeta.assign(sp.outC, 0.0f);
			cl.v2BnBeta.assign(sp.outC, 0.0f);
		}

		initGlorot(rngEngine, cl.W, sp.im2colCols, sp.outC);

		if (sp.useMaxPool)
		{
			curH = sp.poolOutH; curW = sp.poolOutW;
		}
		else
		{
			curH = sp.outH; curW = sp.outW;
		}
		curC = sp.outC;
	}

	cs.flattenedSize = curC * curH * curW;

	// FC layers from NNInfo hidden layers
	const int H = skel->numHiddenLayers();
	unsigned int fcIn = cs.flattenedSize;

	for (int l = 0; l < H; ++l)
	{
		const int hs = skel->getHiddenLayerSize(static_cast<unsigned int>(l));
		const unsigned int fcOut = (hs > 0) ? static_cast<unsigned int>(hs) : 0u;
		if (fcOut == 0u)
			return false;

		NNetwork::TensorCNNState::FCTransition fc;
		fc.in = fcIn;
		fc.out = fcOut;
		fc.W.assign(static_cast<size_t>(fcOut) * fcIn, 0.0f);
		fc.bias.assign(fcOut, 0.0f);
		fc.gW.assign(fc.W.size(), 0.0f);
		fc.gBias.assign(fcOut, 0.0f);
		fc.vW.assign(fc.W.size(), 0.0f);
		fc.v2W.assign(fc.W.size(), 0.0f);
		fc.vBias.assign(fcOut, 0.0f);
		fc.v2Bias.assign(fcOut, 0.0f);

		initGlorot(rngEngine, fc.W, fcIn, fcOut);
		cs.fcLayers.push_back(fc);
		fcIn = fcOut;
	}

	// Final FC layer to outputSize
	{
		NNetwork::TensorCNNState::FCTransition fc;
		fc.in = fcIn;
		fc.out = outputSize;
		fc.W.assign(static_cast<size_t>(outputSize) * fcIn, 0.0f);
		fc.bias.assign(outputSize, 0.0f);
		fc.gW.assign(fc.W.size(), 0.0f);
		fc.gBias.assign(outputSize, 0.0f);
		fc.vW.assign(fc.W.size(), 0.0f);
		fc.v2W.assign(fc.W.size(), 0.0f);
		fc.vBias.assign(outputSize, 0.0f);
		fc.v2Bias.assign(outputSize, 0.0f);

		initGlorot(rngEngine, fc.W, fcIn, outputSize);
		cs.fcLayers.push_back(fc);
	}

	cs.optimizerStep = 0ULL;
	cs.batchCount = 0u;
	cs.initialized = true;
	return true;
}

// ============================================================
// Adam state initialization
// ============================================================

void GAN::initAdamState(AdamState& state, const NNetwork& net)
{
	state.step = 0ULL;

	// Check actual tensor state (not config.archType) because the generator
	// always uses DFF even when the discriminator uses CNN.
	if (net.tensorDff.initialized)
	{
		const unsigned int numT = static_cast<unsigned int>(net.tensorDff.T.size());
		state.mW.resize(numT);
		state.vW.resize(numT);
		state.mBias.resize(numT);
		state.vBias.resize(numT);
		for (unsigned int t = 0; t < numT; ++t)
		{
			const size_t wSz = net.tensorDff.T[t].W.size();
			const size_t bSz = net.tensorDff.T[t].bias.size();
			state.mW[t].assign(wSz, 0.0f);
			state.vW[t].assign(wSz, 0.0f);
			state.mBias[t].assign(bSz, 0.0f);
			state.vBias[t].assign(bSz, 0.0f);
		}
	}
	if (net.tensorCnn.initialized)
	{
		const NNetwork::TensorCNNState& cs = net.tensorCnn;
		const unsigned int numConv = static_cast<unsigned int>(cs.convLayers.size());
		state.mCW.resize(numConv);
		state.vCW.resize(numConv);
		state.mCBias.resize(numConv);
		state.vCBias.resize(numConv);
		for (unsigned int l = 0; l < numConv; ++l)
		{
			state.mCW[l].assign(cs.convLayers[l].W.size(), 0.0f);
			state.vCW[l].assign(cs.convLayers[l].W.size(), 0.0f);
			state.mCBias[l].assign(cs.convLayers[l].bias.size(), 0.0f);
			state.vCBias[l].assign(cs.convLayers[l].bias.size(), 0.0f);
		}
		const unsigned int numFC = static_cast<unsigned int>(cs.fcLayers.size());
		state.mFW.resize(numFC);
		state.vFW.resize(numFC);
		state.mFBias.resize(numFC);
		state.vFBias.resize(numFC);
		for (unsigned int t = 0; t < numFC; ++t)
		{
			state.mFW[t].assign(cs.fcLayers[t].W.size(), 0.0f);
			state.vFW[t].assign(cs.fcLayers[t].W.size(), 0.0f);
			state.mFBias[t].assign(cs.fcLayers[t].bias.size(), 0.0f);
			state.vFBias[t].assign(cs.fcLayers[t].bias.size(), 0.0f);
		}
	}
}

// ============================================================
// Layer Normalization helpers
// ============================================================

void GAN::initLayerNorm(LayerNormParams& lnp, const NNetwork& net)
{
	if (lnp.initialized)
		return;
	if (!net.tensorDff.initialized)
		return;

	const unsigned int numT = static_cast<unsigned int>(net.tensorDff.T.size());
	if (numT < 2u)
		return; // need at least one hidden layer

	// LN is applied to all transitions except the last one (output layer)
	const unsigned int numLN = numT - 1u;
	lnp.gamma.resize(numLN);
	lnp.beta.resize(numLN);
	lnp.gGamma.resize(numLN);
	lnp.gBeta.resize(numLN);
	lnp.mGamma.resize(numLN);
	lnp.vGamma.resize(numLN);
	lnp.mBeta.resize(numLN);
	lnp.vBeta.resize(numLN);
	lnp.zNorm.resize(numLN);
	lnp.invStd.resize(numLN, 0.0f);

	for (unsigned int t = 0; t < numLN; ++t)
	{
		const unsigned int outSz = net.tensorDff.T[t].out;
		lnp.gamma[t].assign(outSz, 1.0f);  // init to 1 (identity)
		lnp.beta[t].assign(outSz, 0.0f);   // init to 0
		lnp.gGamma[t].assign(outSz, 0.0f);
		lnp.gBeta[t].assign(outSz, 0.0f);
		lnp.mGamma[t].assign(outSz, 0.0f);
		lnp.vGamma[t].assign(outSz, 0.0f);
		lnp.mBeta[t].assign(outSz, 0.0f);
		lnp.vBeta[t].assign(outSz, 0.0f);
		lnp.zNorm[t].assign(outSz, 0.0f);
	}

	lnp.step = 0ULL;
	lnp.initialized = true;
}

void GAN::scaleLayerNormGrads(LayerNormParams& lnp, float scale)
{
	if (!lnp.initialized) return;
	for (unsigned int t = 0; t < lnp.gGamma.size(); ++t)
	{
		for (size_t i = 0; i < lnp.gGamma[t].size(); ++i)
			lnp.gGamma[t][i] *= scale;
		for (size_t i = 0; i < lnp.gBeta[t].size(); ++i)
			lnp.gBeta[t][i] *= scale;
	}
}

void GAN::layerNormUpdate(LayerNormParams& lnp, float lr)
{
	if (!lnp.initialized) return;
	++lnp.step;
	const float beta1 = config.adamBeta1;
	const float beta2 = config.adamBeta2;
	const float eps = config.adamEps;
	const float bc1 = 1.0f - powf(beta1, static_cast<float>(lnp.step));
	const float bc2 = 1.0f - powf(beta2, static_cast<float>(lnp.step));

	for (unsigned int t = 0; t < lnp.gamma.size(); ++t)
	{
		for (size_t i = 0; i < lnp.gamma[t].size(); ++i)
		{
			const float gG = lnp.gGamma[t][i];
			lnp.mGamma[t][i] = beta1 * lnp.mGamma[t][i] + (1.0f - beta1) * gG;
			lnp.vGamma[t][i] = beta2 * lnp.vGamma[t][i] + (1.0f - beta2) * gG * gG;
			const float mHat = lnp.mGamma[t][i] / bc1;
			const float vHat = lnp.vGamma[t][i] / bc2;
			lnp.gamma[t][i] -= lr * mHat / (sqrtf(vHat) + eps);
			lnp.gGamma[t][i] = 0.0f;
		}
		for (size_t i = 0; i < lnp.beta[t].size(); ++i)
		{
			const float gB = lnp.gBeta[t][i];
			lnp.mBeta[t][i] = beta1 * lnp.mBeta[t][i] + (1.0f - beta1) * gB;
			lnp.vBeta[t][i] = beta2 * lnp.vBeta[t][i] + (1.0f - beta2) * gB * gB;
			const float mHat = lnp.mBeta[t][i] / bc1;
			const float vHat = lnp.vBeta[t][i] / bc2;
			lnp.beta[t][i] -= lr * mHat / (sqrtf(vHat) + eps);
			lnp.gBeta[t][i] = 0.0f;
		}
	}
}

GAN::LayerNormParams* GAN::genLN() const
{
	return config.generatorLayerNorm ? &genLNParams : NULL;
}

GAN::LayerNormParams* GAN::genBALN() const
{
	return config.generatorLayerNorm ? &genBALNParams : NULL;
}

// ============================================================
// DFF Forward Pass
// ============================================================

void GAN::dffForward(const NNetwork& net, const float* input, unsigned int inputSize,
                     std::vector<std::vector<float> >& activations,
                     bool sigmoidOutput,
                     LayerNormParams* lnp) const
{
	const NNetwork::TensorDFFState& dff = net.tensorDff;
	const NNInfo* skel = net.skeleton;
	const unsigned int numLayers = static_cast<unsigned int>(dff.sizes.size());

	activations.resize(numLayers);
	activations[0].assign(input, input + inputSize);

	for (unsigned int t = 0; t < dff.T.size(); ++t)
	{
		const NNetwork::TensorDFFState::Transition& tr = dff.T[t];
		const int actFx = skel->getActivationType(t);
		const float actParam = skel->getActivationParam(t);
		const bool isLastLayer = (t == dff.T.size() - 1u);
		const bool applyLN = (lnp != NULL && lnp->initialized && !isLastLayer && t < lnp->gamma.size());

		activations[t + 1u].assign(tr.out, 0.0f);

		// Step 1: linear transform (SIMD GEMV)
		transformer_kernels::gemv_rowmajor_bias_block4_unroll8_into(
			activations[t].data(), tr.in,
			tr.W.data(), tr.out,
			tr.bias.data(), tr.out,
			&activations[t + 1u][0]);

		// Step 2: layer normalization (hidden layers only)
		if (applyLN)
		{
			const unsigned int n = tr.out;
			float mu = 0.0f;
			for (unsigned int j = 0; j < n; ++j)
				mu += activations[t + 1u][j];
			mu /= static_cast<float>(n);

			float var = 0.0f;
			for (unsigned int j = 0; j < n; ++j)
			{
				const float d = activations[t + 1u][j] - mu;
				var += d * d;
			}
			var /= static_cast<float>(n);

			const float invStd = 1.0f / sqrtf(var + 1e-5f);
			lnp->invStd[t] = invStd;

			for (unsigned int j = 0; j < n; ++j)
			{
				const float zn = (activations[t + 1u][j] - mu) * invStd;
				lnp->zNorm[t][j] = zn;
				activations[t + 1u][j] = lnp->gamma[t][j] * zn + lnp->beta[t][j];
			}
		}

		// Step 3: activation
		if (isLastLayer && sigmoidOutput)
		{
			for (unsigned int j = 0; j < tr.out; ++j)
				activations[t + 1u][j] = sigmoid(activations[t + 1u][j]);
		}
		else
		{
			for (unsigned int j = 0; j < tr.out; ++j)
				activations[t + 1u][j] = GMath::squash(activations[t + 1u][j], actFx, actParam);
		}
	}
}

// ============================================================
// DFF Backward Pass (computes input gradients too)
// ============================================================

void GAN::dffBackward(NNetwork& net, const std::vector<std::vector<float> >& activations,
                      const float* outputGrad, unsigned int outputSize,
                      std::vector<float>* inputGrad,
                      bool sigmoidOutput,
                      LayerNormParams* lnp)
{
	NNetwork::TensorDFFState& dff = net.tensorDff;
	const NNInfo* skel = net.skeleton;
	const unsigned int numLayers = static_cast<unsigned int>(dff.sizes.size());
	if (numLayers < 2u)
		return;

	const unsigned int lastLayer = numLayers - 1u;

	// Compute deltas - reuse class scratch buffer
	std::vector<std::vector<float> >& delta = scratchDelta;
	delta.resize(numLayers);
	for (unsigned int li = 0; li < numLayers; ++li)
		delta[li].assign(dff.sizes[li], 0.0f);

	// Output deltas: outputGrad is dL/d(output activation)
	// For sigmoid output, chain rule: dL/dz = dL/da * da/dz = outputGrad * sigmoid'(a)
	for (unsigned int k = 0; k < outputSize; ++k)
	{
		const float a = activations[lastLayer][k];
		const unsigned int lastTransition = static_cast<unsigned int>(dff.T.size() - 1u);

		if (sigmoidOutput)
		{
			// Sigmoid output: da/dz = a*(1-a)
			delta[lastLayer][k] = outputGrad[k] * sigmoidDeriv(a);
		}
		else
		{
			const int actFx = skel->getActivationType(lastTransition);
			const float actParam = skel->getActivationParam(lastTransition);
			const float dA_dZ = GMath::activationErrDer(a, actFx, actParam);
			delta[lastLayer][k] = outputGrad[k] * dA_dZ;
		}
	}

	// Hidden layer deltas (backwards, all the way to layer 1 AND layer 0 for input grad)
	std::vector<float> dLdz_norm;
	const int stopLayer = (inputGrad != NULL) ? 0 : 1;
	for (int li = static_cast<int>(lastLayer) - 1; li >= stopLayer; --li)
	{
		const unsigned int l = static_cast<unsigned int>(li);
		const NNetwork::TensorDFFState::Transition& nextTr = dff.T[l]; // maps layer l -> l+1

		// Check if transition (l-1) used layer norm (it produced activations[l])
		const bool hasLN = (lnp != NULL && lnp->initialized && l > 0u && (l - 1u) < lnp->gamma.size());

		if (hasLN)
		{
			// Two-pass backward through LN: need all dL/da[l] first,
			// then backprop through activation -> LN affine -> normalization
			const unsigned int tLN = l - 1u;
			const unsigned int n = dff.sizes[l];
			const int actFx = skel->getActivationType(tLN);
			const float actParam = skel->getActivationParam(tLN);

			// Pass 1: compute dL/da[l], then through activation and LN affine
			dLdz_norm.assign(n, 0.0f);
			for (unsigned int i = 0; i < n; ++i)
			{
				float sum = 0.0f;
				for (unsigned int j = 0; j < nextTr.out; ++j)
					sum += delta[l + 1u][j] * nextTr.W[static_cast<size_t>(j) * static_cast<size_t>(nextTr.in) + i];

				// Through activation derivative
				const float dA_dZ = GMath::activationErrDer(activations[l][i], actFx, actParam);
				const float dLdz_ln = sum * dA_dZ;

				// Through LN affine: accumulate gamma/beta gradients
				lnp->gGamma[tLN][i] += dLdz_ln * lnp->zNorm[tLN][i];
				lnp->gBeta[tLN][i] += dLdz_ln;
				dLdz_norm[i] = dLdz_ln * lnp->gamma[tLN][i];
			}

			// Pass 2: through normalization
			float mean_dzn = 0.0f, mean_dzn_zn = 0.0f;
			for (unsigned int i = 0; i < n; ++i)
			{
				mean_dzn += dLdz_norm[i];
				mean_dzn_zn += dLdz_norm[i] * lnp->zNorm[tLN][i];
			}
			const float invN = 1.0f / static_cast<float>(n);
			mean_dzn *= invN;
			mean_dzn_zn *= invN;

			for (unsigned int i = 0; i < n; ++i)
			{
				delta[l][i] = lnp->invStd[tLN] *
					(dLdz_norm[i] - mean_dzn - lnp->zNorm[tLN][i] * mean_dzn_zn);
			}
		}
		else
		{
			// Original path (no LN) — row-wise axpy for W^T * delta
			delta[l].assign(dff.sizes[l], 0.0f);
			for (unsigned int j = 0; j < nextTr.out; ++j)
				transformer_kernels::axpy_f32(
					delta[l].data(),
					&nextTr.W[static_cast<size_t>(j) * nextTr.in],
					delta[l + 1u][j], nextTr.in);
			if (l > 0u)
			{
				const int actFx = skel->getActivationType(l - 1u);
				const float actParam = skel->getActivationParam(l - 1u);
				for (unsigned int i = 0; i < dff.sizes[l]; ++i)
					delta[l][i] *= GMath::activationErrDer(activations[l][i], actFx, actParam);
			}
		}
	}

	// Accumulate weight gradients (SIMD axpy)
	for (unsigned int t = 0; t < dff.T.size(); ++t)
	{
		NNetwork::TensorDFFState::Transition& tr = dff.T[t];
		for (unsigned int j = 0; j < tr.out; ++j)
		{
			tr.gBias[j] += delta[t + 1u][j];
			transformer_kernels::axpy_f32(
				&tr.gW[static_cast<size_t>(j) * tr.in],
				activations[t].data(),
				delta[t + 1u][j], tr.in);
		}
	}

	// Return input gradient if requested
	if (inputGrad != NULL)
	{
		inputGrad->assign(delta[0].begin(), delta[0].end());
	}
}

// ============================================================
// DFF Backward Pass (GradientBuffer overload - thread-safe)
// ============================================================

void GAN::dffBackward(const NNetwork& net, const std::vector<std::vector<float> >& activations,
                      const float* outputGrad, unsigned int outputSize,
                      std::vector<float>* inputGrad,
                      GradientBuffer& gradBuf,
                      std::vector<std::vector<float> >& scratchDeltaLocal,
                      bool sigmoidOutput)
{
	const NNetwork::TensorDFFState& dff = net.tensorDff;
	const NNInfo* skel = net.skeleton;
	const unsigned int numLayers = static_cast<unsigned int>(dff.sizes.size());
	if (numLayers < 2u)
		return;

	const unsigned int lastLayer = numLayers - 1u;

	// Compute deltas - use caller-provided scratch buffer
	std::vector<std::vector<float> >& delta = scratchDeltaLocal;
	delta.resize(numLayers);
	for (unsigned int li = 0; li < numLayers; ++li)
		delta[li].assign(dff.sizes[li], 0.0f);

	// Output deltas
	for (unsigned int k = 0; k < outputSize; ++k)
	{
		const float a = activations[lastLayer][k];
		const unsigned int lastTransition = static_cast<unsigned int>(dff.T.size() - 1u);

		if (sigmoidOutput)
		{
			delta[lastLayer][k] = outputGrad[k] * sigmoidDeriv(a);
		}
		else
		{
			const int actFx = skel->getActivationType(lastTransition);
			const float actParam = skel->getActivationParam(lastTransition);
			const float dA_dZ = GMath::activationErrDer(a, actFx, actParam);
			delta[lastLayer][k] = outputGrad[k] * dA_dZ;
		}
	}

	// Hidden layer deltas (no LN support in this overload)
	const int stopLayer = (inputGrad != NULL) ? 0 : 1;
	for (int li = static_cast<int>(lastLayer) - 1; li >= stopLayer; --li)
	{
		const unsigned int l = static_cast<unsigned int>(li);
		const NNetwork::TensorDFFState::Transition& nextTr = dff.T[l];

		// Row-wise axpy for W^T * delta
		delta[l].assign(dff.sizes[l], 0.0f);
		for (unsigned int j = 0; j < nextTr.out; ++j)
			transformer_kernels::axpy_f32(
				delta[l].data(),
				&nextTr.W[static_cast<size_t>(j) * nextTr.in],
				delta[l + 1u][j], nextTr.in);
		if (l > 0u)
		{
			const int actFx = skel->getActivationType(l - 1u);
			const float actParam = skel->getActivationParam(l - 1u);
			for (unsigned int i = 0; i < dff.sizes[l]; ++i)
				delta[l][i] *= GMath::activationErrDer(activations[l][i], actFx, actParam);
		}
	}

	// Accumulate weight gradients into GradientBuffer (SIMD axpy)
	for (unsigned int t = 0; t < dff.T.size(); ++t)
	{
		const NNetwork::TensorDFFState::Transition& tr = dff.T[t];
		for (unsigned int j = 0; j < tr.out; ++j)
		{
			gradBuf.dffGBias[t][j] += delta[t + 1u][j];
			transformer_kernels::axpy_f32(
				&gradBuf.dffGW[t][static_cast<size_t>(j) * tr.in],
				activations[t].data(),
				delta[t + 1u][j], tr.in);
		}
	}

	// Return input gradient if requested
	if (inputGrad != NULL)
	{
		inputGrad->assign(delta[0].begin(), delta[0].end());
	}
}

// ============================================================
// DFF Backward Pass (GradientBuffer + LayerNorm overload - thread-safe)
// ============================================================

void GAN::dffBackward(const NNetwork& net, const std::vector<std::vector<float> >& activations,
                      const float* outputGrad, unsigned int outputSize,
                      std::vector<float>* inputGrad,
                      GradientBuffer& gradBuf,
                      std::vector<std::vector<float> >& scratchDeltaLocal,
                      bool sigmoidOutput,
                      const LayerNormParams* lnp,
                      std::vector<std::vector<float> >& lnGGammaLocal,
                      std::vector<std::vector<float> >& lnGBetaLocal)
{
	const NNetwork::TensorDFFState& dff = net.tensorDff;
	const NNInfo* skel = net.skeleton;
	const unsigned int numLayers = static_cast<unsigned int>(dff.sizes.size());
	if (numLayers < 2u)
		return;

	const unsigned int lastLayer = numLayers - 1u;

	// Compute deltas - use caller-provided scratch buffer
	std::vector<std::vector<float> >& delta = scratchDeltaLocal;
	delta.resize(numLayers);
	for (unsigned int li = 0; li < numLayers; ++li)
		delta[li].assign(dff.sizes[li], 0.0f);

	// Output deltas
	for (unsigned int k = 0; k < outputSize; ++k)
	{
		const float a = activations[lastLayer][k];
		const unsigned int lastTransition = static_cast<unsigned int>(dff.T.size() - 1u);

		if (sigmoidOutput)
		{
			delta[lastLayer][k] = outputGrad[k] * sigmoidDeriv(a);
		}
		else
		{
			const int actFx = skel->getActivationType(lastTransition);
			const float actParam = skel->getActivationParam(lastTransition);
			const float dA_dZ = GMath::activationErrDer(a, actFx, actParam);
			delta[lastLayer][k] = outputGrad[k] * dA_dZ;
		}
	}

	// Hidden layer deltas
	std::vector<float> dLdz_norm;
	const int stopLayer = (inputGrad != NULL) ? 0 : 1;
	for (int li = static_cast<int>(lastLayer) - 1; li >= stopLayer; --li)
	{
		const unsigned int l = static_cast<unsigned int>(li);
		const NNetwork::TensorDFFState::Transition& nextTr = dff.T[l];

		const bool hasLN = (lnp != NULL && lnp->initialized && l > 0u && (l - 1u) < lnp->gamma.size());

		if (hasLN)
		{
			const unsigned int tLN = l - 1u;
			const unsigned int n = dff.sizes[l];
			const int actFx = skel->getActivationType(tLN);
			const float actParam = skel->getActivationParam(tLN);

			// Pass 1: compute dL/da[l], then through activation and LN affine
			dLdz_norm.assign(n, 0.0f);
			for (unsigned int i = 0; i < n; ++i)
			{
				float sum = 0.0f;
				for (unsigned int j = 0; j < nextTr.out; ++j)
					sum += delta[l + 1u][j] * nextTr.W[static_cast<size_t>(j) * static_cast<size_t>(nextTr.in) + i];

				const float dA_dZ = GMath::activationErrDer(activations[l][i], actFx, actParam);
				const float dLdz_ln = sum * dA_dZ;

				// Accumulate gamma/beta gradients into thread-local buffers
				lnGGammaLocal[tLN][i] += dLdz_ln * lnp->zNorm[tLN][i];
				lnGBetaLocal[tLN][i] += dLdz_ln;
				dLdz_norm[i] = dLdz_ln * lnp->gamma[tLN][i];
			}

			// Pass 2: through normalization
			float mean_dzn = 0.0f, mean_dzn_zn = 0.0f;
			for (unsigned int i = 0; i < n; ++i)
			{
				mean_dzn += dLdz_norm[i];
				mean_dzn_zn += dLdz_norm[i] * lnp->zNorm[tLN][i];
			}
			const float invN = 1.0f / static_cast<float>(n);
			mean_dzn *= invN;
			mean_dzn_zn *= invN;

			for (unsigned int i = 0; i < n; ++i)
			{
				delta[l][i] = lnp->invStd[tLN] *
					(dLdz_norm[i] - mean_dzn - lnp->zNorm[tLN][i] * mean_dzn_zn);
			}
		}
		else
		{
			// Original path (no LN) - row-wise axpy for W^T * delta
			delta[l].assign(dff.sizes[l], 0.0f);
			for (unsigned int j = 0; j < nextTr.out; ++j)
				transformer_kernels::axpy_f32(
					delta[l].data(),
					&nextTr.W[static_cast<size_t>(j) * nextTr.in],
					delta[l + 1u][j], nextTr.in);
			if (l > 0u)
			{
				const int actFx = skel->getActivationType(l - 1u);
				const float actParam = skel->getActivationParam(l - 1u);
				for (unsigned int i = 0; i < dff.sizes[l]; ++i)
					delta[l][i] *= GMath::activationErrDer(activations[l][i], actFx, actParam);
			}
		}
	}

	// Accumulate weight gradients into GradientBuffer (SIMD axpy)
	for (unsigned int t = 0; t < dff.T.size(); ++t)
	{
		const NNetwork::TensorDFFState::Transition& tr = dff.T[t];
		for (unsigned int j = 0; j < tr.out; ++j)
		{
			gradBuf.dffGBias[t][j] += delta[t + 1u][j];
			transformer_kernels::axpy_f32(
				&gradBuf.dffGW[t][static_cast<size_t>(j) * tr.in],
				activations[t].data(),
				delta[t + 1u][j], tr.in);
		}
	}

	// Return input gradient if requested
	if (inputGrad != NULL)
	{
		inputGrad->assign(delta[0].begin(), delta[0].end());
	}
}

// ============================================================
// DFF Adam Update
// ============================================================

void GAN::dffUpdate(NNetwork& net, AdamState& state, float lr)
{
	NNetwork::TensorDFFState& dff = net.tensorDff;
	++state.step;
	const float beta1 = config.adamBeta1;
	const float beta2 = config.adamBeta2;
	const float eps = config.adamEps;
	const float bc1 = 1.0f - powf(beta1, static_cast<float>(state.step));
	const float bc2 = 1.0f - powf(beta2, static_cast<float>(state.step));

	for (unsigned int t = 0; t < dff.T.size(); ++t)
	{
		NNetwork::TensorDFFState::Transition& tr = dff.T[t];

		for (size_t i = 0; i < tr.W.size(); ++i)
		{
			const float g = tr.gW[i];
			state.mW[t][i] = beta1 * state.mW[t][i] + (1.0f - beta1) * g;
			state.vW[t][i] = beta2 * state.vW[t][i] + (1.0f - beta2) * g * g;
			const float mHat = state.mW[t][i] / bc1;
			const float vHat = state.vW[t][i] / bc2;
			tr.W[i] -= lr * mHat / (sqrtf(vHat) + eps);
			tr.gW[i] = 0.0f;
		}

		for (unsigned int j = 0; j < tr.out; ++j)
		{
			const float g = tr.gBias[j];
			state.mBias[t][j] = beta1 * state.mBias[t][j] + (1.0f - beta1) * g;
			state.vBias[t][j] = beta2 * state.vBias[t][j] + (1.0f - beta2) * g * g;
			const float mHat = state.mBias[t][j] / bc1;
			const float vHat = state.vBias[t][j] / bc2;
			tr.bias[j] -= lr * mHat / (sqrtf(vHat) + eps);
			tr.gBias[j] = 0.0f;
		}
	}
}

// ============================================================
// CNN Forward Pass
// ============================================================

void GAN::cnnForward(const NNetwork& net, const float* input,
                     std::vector<float>& output,
                     bool sigmoidOutput,
                     std::vector<std::vector<float> >* fcActivations) const
{
	const NNetwork::TensorCNNState& cs = net.tensorCnn;
	const unsigned int numConv = static_cast<unsigned int>(cs.convLayers.size());
	const unsigned int numFC = static_cast<unsigned int>(cs.fcLayers.size());

	std::vector<GANConvScratch> convScratch(numConv);

	const float* curInput = input;

	for (unsigned int l = 0; l < numConv; ++l)
	{
		const NNetwork::TensorCNNState::ConvSpatialInfo& sp = cs.spatialInfo[l];
		const NNetwork::TensorCNNState::ConvLayer& cl = cs.convLayers[l];
		GANConvScratch& sc = convScratch[l];
		const unsigned int N = sp.outH * sp.outW;
		const unsigned int K = sp.im2colCols;

		sc.im2col.assign(static_cast<size_t>(sp.im2colRows) * K, 0.0f);
		sc.convOut.assign(static_cast<size_t>(sp.outC) * N, 0.0f);
		sc.actOut.assign(static_cast<size_t>(sp.outC) * N, 0.0f);

		im2col_cpu(curInput, sp.inC, sp.inH, sp.inW,
		           sp.kH, sp.kW, sp.strideH, sp.strideW, sp.padH, sp.padW,
		           sp.outH, sp.outW, &sc.im2col[0]);

		// Bias init + GEMM
		for (unsigned int c = 0; c < sp.outC; ++c)
		{
			const float b = cl.bias[c];
			for (unsigned int n = 0; n < N; ++n)
				sc.convOut[c * N + n] = b;
		}
		sgemm_abt_cpu(&cl.W[0], &sc.im2col[0], &sc.convOut[0], sp.outC, K, N);

		// LeakyReLU (good for discriminators) or ReLU
		for (size_t i = 0; i < static_cast<size_t>(sp.outC) * N; ++i)
			sc.actOut[i] = (sc.convOut[i] > 0.0f) ? sc.convOut[i] : 0.2f * sc.convOut[i];

		if (sp.useMaxPool)
		{
			const size_t poolN = static_cast<size_t>(sp.outC) * sp.poolOutH * sp.poolOutW;
			sc.poolOut.assign(poolN, 0.0f);
			sc.poolArgmax.assign(poolN, 0);
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

	// Flatten + FC layers
	std::vector<std::vector<float> > fcA(numFC + 1u);
	fcA[0].assign(curInput, curInput + cs.flattenedSize);

	for (unsigned int t = 0; t < numFC; ++t)
	{
		const NNetwork::TensorCNNState::FCTransition& fc = cs.fcLayers[t];
		const bool isLast = (t == numFC - 1u);
		fcA[t + 1u].assign(fc.out, 0.0f);

		for (unsigned int j = 0; j < fc.out; ++j)
		{
			float z = fc.bias[j];
			const size_t rowOff = static_cast<size_t>(j) * fc.in;
			for (unsigned int i = 0; i < fc.in; ++i)
				z += fc.W[rowOff + i] * fcA[t][i];

			if (isLast && sigmoidOutput)
				fcA[t + 1u][j] = sigmoid(z);
			else
				fcA[t + 1u][j] = (z > 0.0f) ? z : 0.2f * z; // LeakyReLU
		}
	}

	output = fcA[numFC];
	if (fcActivations)
		*fcActivations = fcA;
}

// ============================================================
// CNN Backward Pass
// ============================================================

void GAN::cnnBackward(NNetwork& net, const float* input,
                      const float* outputGrad, unsigned int outputSize,
                      std::vector<float>* inputGrad,
                      const std::vector<float>* penultGrad,
                      bool sigmoidOutput)
{
	NNetwork::TensorCNNState& cs = net.tensorCnn;
	const unsigned int numConv = static_cast<unsigned int>(cs.convLayers.size());
	const unsigned int numFC = static_cast<unsigned int>(cs.fcLayers.size());

	// Re-do forward pass to get activations (needed for backward)
	std::vector<GANConvScratch> convScratch(numConv);

	const float* curInput = input;

	for (unsigned int l = 0; l < numConv; ++l)
	{
		const NNetwork::TensorCNNState::ConvSpatialInfo& sp = cs.spatialInfo[l];
		const NNetwork::TensorCNNState::ConvLayer& cl = cs.convLayers[l];
		GANConvScratch& sc = convScratch[l];
		const unsigned int N = sp.outH * sp.outW;
		const unsigned int K = sp.im2colCols;

		sc.im2col.assign(static_cast<size_t>(sp.im2colRows) * K, 0.0f);
		sc.convOut.assign(static_cast<size_t>(sp.outC) * N, 0.0f);
		sc.actOut.assign(static_cast<size_t>(sp.outC) * N, 0.0f);

		im2col_cpu(curInput, sp.inC, sp.inH, sp.inW,
		           sp.kH, sp.kW, sp.strideH, sp.strideW, sp.padH, sp.padW,
		           sp.outH, sp.outW, &sc.im2col[0]);

		for (unsigned int c = 0; c < sp.outC; ++c)
		{
			const float b = cl.bias[c];
			for (unsigned int n = 0; n < N; ++n)
				sc.convOut[c * N + n] = b;
		}
		sgemm_abt_cpu(&cl.W[0], &sc.im2col[0], &sc.convOut[0], sp.outC, K, N);

		for (size_t i = 0; i < static_cast<size_t>(sp.outC) * N; ++i)
			sc.actOut[i] = (sc.convOut[i] > 0.0f) ? sc.convOut[i] : 0.2f * sc.convOut[i];

		if (sp.useMaxPool)
		{
			const size_t poolN = static_cast<size_t>(sp.outC) * sp.poolOutH * sp.poolOutW;
			sc.poolOut.assign(poolN, 0.0f);
			sc.poolArgmax.assign(poolN, 0);
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

	// FC forward for activations
	std::vector<std::vector<float> > fcA(numFC + 1u);
	fcA[0].assign(curInput, curInput + cs.flattenedSize);

	for (unsigned int t = 0; t < numFC; ++t)
	{
		const NNetwork::TensorCNNState::FCTransition& fc = cs.fcLayers[t];
		const bool isLast = (t == numFC - 1u);
		fcA[t + 1u].assign(fc.out, 0.0f);

		for (unsigned int j = 0; j < fc.out; ++j)
		{
			float z = fc.bias[j];
			const size_t rowOff = static_cast<size_t>(j) * fc.in;
			for (unsigned int i = 0; i < fc.in; ++i)
				z += fc.W[rowOff + i] * fcA[t][i];

			if (isLast && sigmoidOutput)
				fcA[t + 1u][j] = sigmoid(z);
			else
				fcA[t + 1u][j] = (z > 0.0f) ? z : 0.2f * z;
		}
	}

	// FC backward
	std::vector<std::vector<float> > fcDelta(numFC + 1u);
	for (unsigned int t = 0; t <= numFC; ++t)
		fcDelta[t].assign((t < numFC + 1u && t < fcA.size()) ? fcA[t].size() : 0u, 0.0f);

	// Output delta
	for (unsigned int k = 0; k < outputSize; ++k)
	{
		const float a = fcA[numFC][k];
		if (sigmoidOutput)
			fcDelta[numFC][k] = outputGrad[k] * sigmoidDeriv(a);
		else
			fcDelta[numFC][k] = outputGrad[k] * ((a > 0.0f) ? 1.0f : 0.2f);
	}

	// Hidden FC deltas
	for (int li = static_cast<int>(numFC) - 1; li >= 1; --li)
	{
		const unsigned int l = static_cast<unsigned int>(li);
		const NNetwork::TensorCNNState::FCTransition& nextFC = cs.fcLayers[l];
		for (unsigned int i = 0; i < nextFC.in; ++i)
		{
			float sum = 0.0f;
			for (unsigned int j = 0; j < nextFC.out; ++j)
				sum += fcDelta[l + 1u][j] * nextFC.W[static_cast<size_t>(j) * nextFC.in + i];
			const float a = fcA[l][i];
			fcDelta[l][i] = sum * ((a > 0.0f) ? 1.0f : 0.2f);
		}
		// Inject Q-head gradient at penultimate layer
		if (penultGrad && l == numFC - 1u)
		{
			for (unsigned int i = 0; i < nextFC.in && i < penultGrad->size(); ++i)
			{
				const float a = fcA[l][i];
				fcDelta[l][i] += (*penultGrad)[i] * ((a > 0.0f) ? 1.0f : 0.2f);
			}
		}
	}

	// FC gradient accumulation
	for (unsigned int t = 0; t < numFC; ++t)
	{
		NNetwork::TensorCNNState::FCTransition& fc = cs.fcLayers[t];
		for (unsigned int j = 0; j < fc.out; ++j)
		{
			const float d = fcDelta[t + 1u][j];
			fc.gBias[j] += d;
			const size_t rowOff = static_cast<size_t>(j) * fc.in;
			for (unsigned int i = 0; i < fc.in; ++i)
				fc.gW[rowOff + i] += d * fcA[t][i];
		}
	}

	// Gradient from FC to flatten
	std::vector<float> dFlatten(cs.flattenedSize, 0.0f);
	if (numFC > 0u)
	{
		const NNetwork::TensorCNNState::FCTransition& firstFC = cs.fcLayers[0];
		for (unsigned int i = 0; i < firstFC.in; ++i)
		{
			float sum = 0.0f;
			for (unsigned int j = 0; j < firstFC.out; ++j)
				sum += fcDelta[1u][j] * firstFC.W[static_cast<size_t>(j) * firstFC.in + i];
			dFlatten[i] = sum;
		}
	}

	// Conv backward
	const float* dUpstream = &dFlatten[0];
	std::vector<float> convInputGradBuf;

	for (int li = static_cast<int>(numConv) - 1; li >= 0; --li)
	{
		const unsigned int l = static_cast<unsigned int>(li);
		const NNetwork::TensorCNNState::ConvSpatialInfo& sp = cs.spatialInfo[l];
		NNetwork::TensorCNNState::ConvLayer& cl = cs.convLayers[l];
		GANConvScratch& sc = convScratch[l];
		const unsigned int N = sp.outH * sp.outW;
		const unsigned int K = sp.im2colCols;

		std::vector<float> dActOut(static_cast<size_t>(sp.outC) * N, 0.0f);

		// MaxPool backward
		if (sp.useMaxPool)
		{
			maxpool_backward_cpu(dUpstream, &sc.poolArgmax[0],
			                     sp.outC, sp.outH, sp.outW,
			                     sp.poolOutH, sp.poolOutW,
			                     &dActOut[0]);
		}
		else
		{
			std::memcpy(&dActOut[0], dUpstream, dActOut.size() * sizeof(float));
		}

		// LeakyReLU backward
		for (size_t i = 0; i < dActOut.size(); ++i)
			dActOut[i] = (sc.convOut[i] > 0.0f) ? dActOut[i] : 0.2f * dActOut[i];

		// Weight gradient
		sgemm_cpu(&dActOut[0], &sc.im2col[0], &cl.gW[0], sp.outC, N, K);

		// Bias gradient
		for (unsigned int c = 0; c < sp.outC; ++c)
		{
			float bsum = 0.0f;
			for (unsigned int n = 0; n < N; ++n)
				bsum += dActOut[c * N + n];
			cl.gBias[c] += bsum;
		}

		// Input gradient
		if (l > 0u || inputGrad != NULL)
		{
			std::vector<float> dIm2col(static_cast<size_t>(sp.im2colRows) * K, 0.0f);
			sgemm_atb_cpu(&dActOut[0], &cl.W[0], &dIm2col[0], sp.outC, N, K);

			const size_t inputVol = static_cast<size_t>(sp.inC) * sp.inH * sp.inW;
			convInputGradBuf.assign(inputVol, 0.0f);
			col2im_cpu(&dIm2col[0], sp.inC, sp.inH, sp.inW,
			           sp.kH, sp.kW, sp.strideH, sp.strideW, sp.padH, sp.padW,
			           sp.outH, sp.outW, &convInputGradBuf[0]);
			dUpstream = &convInputGradBuf[0];
		}
	}

	if (inputGrad != NULL && !convInputGradBuf.empty())
		*inputGrad = convInputGradBuf;
}

// ============================================================
// CNN Backward Pass (GradientBuffer overload - thread-safe)
// ============================================================

void GAN::cnnBackward(const NNetwork& net, const float* input,
                      const float* outputGrad, unsigned int outputSize,
                      std::vector<float>* inputGrad,
                      GradientBuffer& gradBuf,
                      const std::vector<float>* penultGrad,
                      bool sigmoidOutput)
{
	const NNetwork::TensorCNNState& cs = net.tensorCnn;
	const unsigned int numConv = static_cast<unsigned int>(cs.convLayers.size());
	const unsigned int numFC = static_cast<unsigned int>(cs.fcLayers.size());

	// Re-do forward pass to get activations (needed for backward)
	std::vector<GANConvScratch> convScratch(numConv);

	const float* curInput = input;

	for (unsigned int l = 0; l < numConv; ++l)
	{
		const NNetwork::TensorCNNState::ConvSpatialInfo& sp = cs.spatialInfo[l];
		const NNetwork::TensorCNNState::ConvLayer& cl = cs.convLayers[l];
		GANConvScratch& sc = convScratch[l];
		const unsigned int N = sp.outH * sp.outW;
		const unsigned int K = sp.im2colCols;

		sc.im2col.assign(static_cast<size_t>(sp.im2colRows) * K, 0.0f);
		sc.convOut.assign(static_cast<size_t>(sp.outC) * N, 0.0f);
		sc.actOut.assign(static_cast<size_t>(sp.outC) * N, 0.0f);

		im2col_cpu(curInput, sp.inC, sp.inH, sp.inW,
		           sp.kH, sp.kW, sp.strideH, sp.strideW, sp.padH, sp.padW,
		           sp.outH, sp.outW, &sc.im2col[0]);

		for (unsigned int c = 0; c < sp.outC; ++c)
		{
			const float b = cl.bias[c];
			for (unsigned int n = 0; n < N; ++n)
				sc.convOut[c * N + n] = b;
		}
		sgemm_abt_cpu(&cl.W[0], &sc.im2col[0], &sc.convOut[0], sp.outC, K, N);

		for (size_t i = 0; i < static_cast<size_t>(sp.outC) * N; ++i)
			sc.actOut[i] = (sc.convOut[i] > 0.0f) ? sc.convOut[i] : 0.2f * sc.convOut[i];

		if (sp.useMaxPool)
		{
			const size_t poolN = static_cast<size_t>(sp.outC) * sp.poolOutH * sp.poolOutW;
			sc.poolOut.assign(poolN, 0.0f);
			sc.poolArgmax.assign(poolN, 0);
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

	// FC forward for activations
	std::vector<std::vector<float> > fcA(numFC + 1u);
	fcA[0].assign(curInput, curInput + cs.flattenedSize);

	for (unsigned int t = 0; t < numFC; ++t)
	{
		const NNetwork::TensorCNNState::FCTransition& fc = cs.fcLayers[t];
		const bool isLast = (t == numFC - 1u);
		fcA[t + 1u].assign(fc.out, 0.0f);

		for (unsigned int j = 0; j < fc.out; ++j)
		{
			float z = fc.bias[j];
			const size_t rowOff = static_cast<size_t>(j) * fc.in;
			for (unsigned int i = 0; i < fc.in; ++i)
				z += fc.W[rowOff + i] * fcA[t][i];

			if (isLast && sigmoidOutput)
				fcA[t + 1u][j] = sigmoid(z);
			else
				fcA[t + 1u][j] = (z > 0.0f) ? z : 0.2f * z;
		}
	}

	// FC backward
	std::vector<std::vector<float> > fcDelta(numFC + 1u);
	for (unsigned int t = 0; t <= numFC; ++t)
		fcDelta[t].assign((t < numFC + 1u && t < fcA.size()) ? fcA[t].size() : 0u, 0.0f);

	// Output delta
	for (unsigned int k = 0; k < outputSize; ++k)
	{
		const float a = fcA[numFC][k];
		if (sigmoidOutput)
			fcDelta[numFC][k] = outputGrad[k] * sigmoidDeriv(a);
		else
			fcDelta[numFC][k] = outputGrad[k] * ((a > 0.0f) ? 1.0f : 0.2f);
	}

	// Hidden FC deltas
	for (int li = static_cast<int>(numFC) - 1; li >= 1; --li)
	{
		const unsigned int l = static_cast<unsigned int>(li);
		const NNetwork::TensorCNNState::FCTransition& nextFC = cs.fcLayers[l];
		for (unsigned int i = 0; i < nextFC.in; ++i)
		{
			float sum = 0.0f;
			for (unsigned int j = 0; j < nextFC.out; ++j)
				sum += fcDelta[l + 1u][j] * nextFC.W[static_cast<size_t>(j) * nextFC.in + i];
			const float a = fcA[l][i];
			fcDelta[l][i] = sum * ((a > 0.0f) ? 1.0f : 0.2f);
		}
		// Inject Q-head gradient at penultimate layer
		if (penultGrad && l == numFC - 1u)
		{
			for (unsigned int i = 0; i < nextFC.in && i < penultGrad->size(); ++i)
			{
				const float a = fcA[l][i];
				fcDelta[l][i] += (*penultGrad)[i] * ((a > 0.0f) ? 1.0f : 0.2f);
			}
		}
	}

	// FC gradient accumulation into GradientBuffer
	for (unsigned int t = 0; t < numFC; ++t)
	{
		const NNetwork::TensorCNNState::FCTransition& fc = cs.fcLayers[t];
		for (unsigned int j = 0; j < fc.out; ++j)
		{
			const float d = fcDelta[t + 1u][j];
			gradBuf.fcGBias[t][j] += d;
			const size_t rowOff = static_cast<size_t>(j) * fc.in;
			for (unsigned int i = 0; i < fc.in; ++i)
				gradBuf.fcGW[t][rowOff + i] += d * fcA[t][i];
		}
	}

	// Gradient from FC to flatten
	std::vector<float> dFlatten(cs.flattenedSize, 0.0f);
	if (numFC > 0u)
	{
		const NNetwork::TensorCNNState::FCTransition& firstFC = cs.fcLayers[0];
		for (unsigned int i = 0; i < firstFC.in; ++i)
		{
			float sum = 0.0f;
			for (unsigned int j = 0; j < firstFC.out; ++j)
				sum += fcDelta[1u][j] * firstFC.W[static_cast<size_t>(j) * firstFC.in + i];
			dFlatten[i] = sum;
		}
	}

	// Conv backward
	const float* dUpstream = &dFlatten[0];
	std::vector<float> convInputGradBuf;

	for (int li = static_cast<int>(numConv) - 1; li >= 0; --li)
	{
		const unsigned int l = static_cast<unsigned int>(li);
		const NNetwork::TensorCNNState::ConvSpatialInfo& sp = cs.spatialInfo[l];
		const NNetwork::TensorCNNState::ConvLayer& cl = cs.convLayers[l];
		GANConvScratch& sc = convScratch[l];
		const unsigned int N = sp.outH * sp.outW;
		const unsigned int K = sp.im2colCols;

		std::vector<float> dActOut(static_cast<size_t>(sp.outC) * N, 0.0f);

		// MaxPool backward
		if (sp.useMaxPool)
		{
			maxpool_backward_cpu(dUpstream, &sc.poolArgmax[0],
			                     sp.outC, sp.outH, sp.outW,
			                     sp.poolOutH, sp.poolOutW,
			                     &dActOut[0]);
		}
		else
		{
			std::memcpy(&dActOut[0], dUpstream, dActOut.size() * sizeof(float));
		}

		// LeakyReLU backward
		for (size_t i = 0; i < dActOut.size(); ++i)
			dActOut[i] = (sc.convOut[i] > 0.0f) ? dActOut[i] : 0.2f * dActOut[i];

		// Weight gradient into GradientBuffer
		sgemm_cpu(&dActOut[0], &sc.im2col[0], &gradBuf.convGW[l][0], sp.outC, N, K);

		// Bias gradient into GradientBuffer
		for (unsigned int c = 0; c < sp.outC; ++c)
		{
			float bsum = 0.0f;
			for (unsigned int n = 0; n < N; ++n)
				bsum += dActOut[c * N + n];
			gradBuf.convGBias[l][c] += bsum;
		}

		// Input gradient
		if (l > 0u || inputGrad != NULL)
		{
			std::vector<float> dIm2col(static_cast<size_t>(sp.im2colRows) * K, 0.0f);
			sgemm_atb_cpu(&dActOut[0], &cl.W[0], &dIm2col[0], sp.outC, N, K);

			const size_t inputVol = static_cast<size_t>(sp.inC) * sp.inH * sp.inW;
			convInputGradBuf.assign(inputVol, 0.0f);
			col2im_cpu(&dIm2col[0], sp.inC, sp.inH, sp.inW,
			           sp.kH, sp.kW, sp.strideH, sp.strideW, sp.padH, sp.padW,
			           sp.outH, sp.outW, &convInputGradBuf[0]);
			dUpstream = &convInputGradBuf[0];
		}
	}

	if (inputGrad != NULL && !convInputGradBuf.empty())
		*inputGrad = convInputGradBuf;
}

// ============================================================
// CNN Adam Update
// ============================================================

void GAN::cnnUpdate(NNetwork& net, AdamState& state, float lr)
{
	NNetwork::TensorCNNState& cs = net.tensorCnn;
	++state.step;
	const float beta1 = config.adamBeta1;
	const float beta2 = config.adamBeta2;
	const float eps = config.adamEps;
	const float bc1 = 1.0f - powf(beta1, static_cast<float>(state.step));
	const float bc2 = 1.0f - powf(beta2, static_cast<float>(state.step));

	// Conv layers
	for (unsigned int l = 0; l < cs.convLayers.size(); ++l)
	{
		NNetwork::TensorCNNState::ConvLayer& cl = cs.convLayers[l];
		for (size_t i = 0; i < cl.W.size(); ++i)
		{
			const float g = cl.gW[i];
			state.mCW[l][i] = beta1 * state.mCW[l][i] + (1.0f - beta1) * g;
			state.vCW[l][i] = beta2 * state.vCW[l][i] + (1.0f - beta2) * g * g;
			const float mHat = state.mCW[l][i] / bc1;
			const float vHat = state.vCW[l][i] / bc2;
			cl.W[i] -= lr * mHat / (sqrtf(vHat) + eps);
			cl.gW[i] = 0.0f;
		}
		for (unsigned int c = 0; c < cl.outC; ++c)
		{
			const float g = cl.gBias[c];
			state.mCBias[l][c] = beta1 * state.mCBias[l][c] + (1.0f - beta1) * g;
			state.vCBias[l][c] = beta2 * state.vCBias[l][c] + (1.0f - beta2) * g * g;
			const float mHat = state.mCBias[l][c] / bc1;
			const float vHat = state.vCBias[l][c] / bc2;
			cl.bias[c] -= lr * mHat / (sqrtf(vHat) + eps);
			cl.gBias[c] = 0.0f;
		}
	}

	// FC layers
	for (unsigned int t = 0; t < cs.fcLayers.size(); ++t)
	{
		NNetwork::TensorCNNState::FCTransition& fc = cs.fcLayers[t];
		for (size_t i = 0; i < fc.W.size(); ++i)
		{
			const float g = fc.gW[i];
			state.mFW[t][i] = beta1 * state.mFW[t][i] + (1.0f - beta1) * g;
			state.vFW[t][i] = beta2 * state.vFW[t][i] + (1.0f - beta2) * g * g;
			const float mHat = state.mFW[t][i] / bc1;
			const float vHat = state.vFW[t][i] / bc2;
			fc.W[i] -= lr * mHat / (sqrtf(vHat) + eps);
			fc.gW[i] = 0.0f;
		}
		for (unsigned int j = 0; j < fc.out; ++j)
		{
			const float g = fc.gBias[j];
			state.mFBias[t][j] = beta1 * state.mFBias[t][j] + (1.0f - beta1) * g;
			state.vFBias[t][j] = beta2 * state.vFBias[t][j] + (1.0f - beta2) * g * g;
			const float mHat = state.mFBias[t][j] / bc1;
			const float vHat = state.vFBias[t][j] / bc2;
			fc.bias[j] -= lr * mHat / (sqrtf(vHat) + eps);
			fc.gBias[j] = 0.0f;
		}
	}
}

// ============================================================
// Transposed-Convolution (Deconv) Generator
// ============================================================

bool GAN::initDeconvTensors(NNetwork& net, unsigned int inputDim, const DeconvConfig& cfg)
{
	NNetwork::TensorDeconvState& ds = net.tensorDeconv;
	ds.reset();

	ds.projectC = cfg.projectChannels;
	ds.projectH = cfg.projectH;
	ds.projectW = cfg.projectW;

	// FC projection: inputDim -> projectC * projectH * projectW
	ds.fcIn = inputDim;
	ds.fcOut = ds.projectC * ds.projectH * ds.projectW;
	ds.fcW.assign(static_cast<size_t>(ds.fcOut) * ds.fcIn, 0.0f);
	ds.fcBias.assign(ds.fcOut, 0.0f);
	ds.fcGW.assign(ds.fcW.size(), 0.0f);
	ds.fcGBias.assign(ds.fcOut, 0.0f);
	ds.fcVW.assign(ds.fcW.size(), 0.0f);
	ds.fcV2W.assign(ds.fcW.size(), 0.0f);
	ds.fcVBias.assign(ds.fcOut, 0.0f);
	ds.fcV2Bias.assign(ds.fcOut, 0.0f);
	initGlorot(rngEngine, ds.fcW, ds.fcIn, ds.fcOut);

	// Deconv layers
	unsigned int curC = ds.projectC;
	unsigned int curH = ds.projectH;
	unsigned int curW = ds.projectW;

	ds.layers.resize(cfg.layers.size());
	for (size_t li = 0; li < cfg.layers.size(); ++li)
	{
		const DeconvConfig::DeconvLayerSpec& spec = cfg.layers[li];
		NNetwork::TensorDeconvState::DeconvLayer& dl = ds.layers[li];

		dl.inC = curC;
		dl.outC = spec.outChannels;
		dl.kH = spec.kernelH;
		dl.kW = spec.kernelW;
		dl.strideH = spec.strideH;
		dl.strideW = spec.strideW;
		dl.padH = spec.padH;
		dl.padW = spec.padW;
		dl.inH = curH;
		dl.inW = curW;
		dl.useBatchNorm = spec.useBatchNorm;
		dl.useReLU = spec.useReLU;
		dl.useTanh = spec.useTanh;
		dl.useUpsampleConv = spec.useUpsampleConv;

		if (dl.useUpsampleConv)
		{
			// Upsample + standard conv: upH = inH*stride, outH = (upH - kH + 2*pad) + 1
			const unsigned int upH = curH * spec.strideH;
			const unsigned int upW = curW * spec.strideW;
			dl.outH = upH - spec.kernelH + 2u * spec.padH + 1u;
			dl.outW = upW - spec.kernelW + 2u * spec.padW + 1u;
		}
		else
		{
			// Transposed conv: outH = (inH - 1)*stride - 2*pad + kH
			dl.outH = (curH - 1u) * spec.strideH - 2u * spec.padH + spec.kernelH;
			dl.outW = (curW - 1u) * spec.strideW - 2u * spec.padW + spec.kernelW;
		}

		// W: [inC, outC*kH*kW]
		const size_t wSize = static_cast<size_t>(dl.inC) * dl.outC * dl.kH * dl.kW;
		dl.W.assign(wSize, 0.0f);
		dl.bias.assign(dl.outC, 0.0f);
		dl.gW.assign(wSize, 0.0f);
		dl.gBias.assign(dl.outC, 0.0f);
		dl.vW.assign(wSize, 0.0f);
		dl.v2W.assign(wSize, 0.0f);
		dl.vBias.assign(dl.outC, 0.0f);
		dl.v2Bias.assign(dl.outC, 0.0f);

		if (dl.useUpsampleConv)
			initGlorot(rngEngine, dl.W, dl.inC * dl.kH * dl.kW, dl.outC);
		else
			initGlorot(rngEngine, dl.W, dl.inC, dl.outC * dl.kH * dl.kW);

		if (dl.useBatchNorm)
		{
			dl.bnGamma.assign(dl.outC, 1.0f);
			dl.bnBeta.assign(dl.outC, 0.0f);
			dl.bnRunMean.assign(dl.outC, 0.0f);
			dl.bnRunVar.assign(dl.outC, 1.0f);
			dl.gBnGamma.assign(dl.outC, 0.0f);
			dl.gBnBeta.assign(dl.outC, 0.0f);
			dl.vBnGamma.assign(dl.outC, 0.0f);
			dl.v2BnGamma.assign(dl.outC, 0.0f);
			dl.vBnBeta.assign(dl.outC, 0.0f);
			dl.v2BnBeta.assign(dl.outC, 0.0f);
		}

		curC = dl.outC;
		curH = dl.outH;
		curW = dl.outW;
	}

	ds.optimizerStep = 0ULL;
	ds.initialized = true;
	return true;
}

void GAN::deconvForward(const NNetwork& net, const float* input, unsigned int inputSize,
                        std::vector<float>& output,
                        std::vector<std::vector<float> >* scratchOut,
                        unsigned int noiseSeed,
                        DeconvScratchArena* arena) const
{
	const NNetwork::TensorDeconvState& ds = net.tensorDeconv;
	const unsigned int numLayers = static_cast<unsigned int>(ds.layers.size());

	// Scratch layout:
	// scratch[0] = original input copy (for FC backward)
	// scratch[1] = FC output (post-activation)
	// scratch[2..numLayers+1] = deconv layer outputs (post-activation)
	// scratch[numLayers+2..2*numLayers+1] = BN scratch (invstd[outC] + x_hat[outVol])
	const unsigned int numScratch = 2u + numLayers;
	std::vector<std::vector<float> > scratch(numScratch + numLayers);

	// Store original input
	scratch[0].assign(input, input + inputSize);

	// FC projection: z = W * input + bias -> LeakyReLU
	scratch[1].assign(ds.fcOut, 0.0f);
	for (unsigned int j = 0; j < ds.fcOut; ++j)
	{
		float z = ds.fcBias[j];
		const size_t rowOff = static_cast<size_t>(j) * ds.fcIn;
		for (unsigned int i = 0; i < ds.fcIn; ++i)
			z += ds.fcW[rowOff + i] * input[i];
		scratch[1][j] = (z > 0.0f) ? z : 0.2f * z; // LeakyReLU
	}

	// Reshape to [projectC, projectH, projectW] — just reinterpret
	const float* curData = &scratch[1][0];

	for (unsigned int li = 0; li < numLayers; ++li)
	{
		const NNetwork::TensorDeconvState::DeconvLayer& dl = ds.layers[li];
		const unsigned int outVol = dl.outC * dl.outH * dl.outW;

		scratch[2u + li].assign(outVol, 0.0f);

		if (dl.useUpsampleConv)
		{
			// Upsample + standard conv path (no checkerboard artifacts)
			const unsigned int upH = dl.inH * dl.strideH;
			const unsigned int upW = dl.inW * dl.strideW;
			const unsigned int K = dl.inC * dl.kH * dl.kW;
			const unsigned int N = dl.outH * dl.outW;

			// Nearest-neighbor upsample
			float* upBuf;
			std::vector<float> upLocal;
			if (arena && arena->initialized) { upBuf = &arena->upsampled[li][0]; }
			else { upLocal.resize(static_cast<size_t>(dl.inC) * upH * upW); upBuf = &upLocal[0]; }
			nn_upsample_cpu(curData, dl.inC, dl.inH, dl.inW, dl.strideH, dl.strideW, upBuf);

			// im2col on upsampled input (stride=1)
			float* colBuf;
			std::vector<float> colLocal;
			if (arena && arena->initialized) { colBuf = &arena->cols[li][0]; }
			else { colLocal.resize(static_cast<size_t>(N) * K); colBuf = &colLocal[0]; }
			im2col_cpu(upBuf, dl.inC, upH, upW,
			           dl.kH, dl.kW, 1u, 1u, dl.padH, dl.padW,
			           dl.outH, dl.outW, colBuf);

			// Standard conv: output[outC, N] = W[outC, K] * cols^T[K, N]
			sgemm_abt_cpu(&dl.W[0], colBuf, &scratch[2u + li][0], dl.outC, K, N);
		}
		else
		{
			// Transposed conv path
			const unsigned int N = dl.inH * dl.inW;
			const unsigned int outK = dl.outC * dl.kH * dl.kW;

			// output_cols[N, outK] = input^T * W
			// W: [inC, outK], input_flat: [inC, N]
			float* ocBuf;
			std::vector<float> ocLocal;
			if (arena && arena->initialized) {
				ocBuf = &arena->outputCols[li][0];
				std::memset(ocBuf, 0, static_cast<size_t>(outK) * N * sizeof(float));
			} else {
				ocLocal.assign(static_cast<size_t>(outK) * N, 0.0f);
				ocBuf = &ocLocal[0];
			}
			sgemm_atb_cpu(curData, &dl.W[0], ocBuf, dl.inC, N, outK);

			// col2im: scatter outputCols to spatial output
			col2im_cpu(ocBuf, dl.outC, dl.outH, dl.outW,
			           dl.kH, dl.kW, dl.strideH, dl.strideW, dl.padH, dl.padW,
			           dl.inH, dl.inW, &scratch[2u + li][0]);
		}

		// Add bias (broadcast per outC channel)
		for (unsigned int c = 0; c < dl.outC; ++c)
		{
			const float b = dl.bias[c];
			const size_t chOff = static_cast<size_t>(c) * dl.outH * dl.outW;
			for (unsigned int p = 0; p < dl.outH * dl.outW; ++p)
				scratch[2u + li][chOff + p] += b;
		}

		// Batch normalization (spatial/instance norm per channel)
		if (dl.useBatchNorm)
		{
			const unsigned int HW = dl.outH * dl.outW;
			const float bnEps = 1e-5f;
			// Store invstd[outC] then x_hat[outVol] for backward pass
			scratch[numScratch + li].resize(static_cast<size_t>(dl.outC) + outVol);
			float* invstdBuf = &scratch[numScratch + li][0];
			float* xhatBuf = &scratch[numScratch + li][dl.outC];

			for (unsigned int c = 0; c < dl.outC; ++c)
			{
				const size_t chOff = static_cast<size_t>(c) * HW;
				// Mean over spatial dims
				float mean = 0.0f;
				for (unsigned int p = 0; p < HW; ++p)
					mean += scratch[2u + li][chOff + p];
				mean /= static_cast<float>(HW);
				// Variance over spatial dims
				float var = 0.0f;
				for (unsigned int p = 0; p < HW; ++p)
				{
					const float d = scratch[2u + li][chOff + p] - mean;
					var += d * d;
				}
				var /= static_cast<float>(HW);
				const float invstd = 1.0f / sqrtf(var + bnEps);
				invstdBuf[c] = invstd;
				// Normalize, scale, shift
				for (unsigned int p = 0; p < HW; ++p)
				{
					const float xh = (scratch[2u + li][chOff + p] - mean) * invstd;
					xhatBuf[chOff + p] = xh;
					scratch[2u + li][chOff + p] = dl.bnGamma[c] * xh + dl.bnBeta[c];
				}
			}
		}

		// Activation: ReLU for hidden layers, sigmoid for last layer
		if (dl.useReLU)
		{
			for (size_t i = 0; i < outVol; ++i)
				scratch[2u + li][i] = (scratch[2u + li][i] > 0.0f) ? scratch[2u + li][i] : 0.0f;

			// Noise injection (training only): add per-pixel Gaussian noise
			// after hidden-layer activations to prevent mode collapse.
			// Gradient of addition is identity, so no backward changes needed.
			// noiseSeed != 0 indicates training mode; seed is unique per thread/sample.
			if (noiseSeed != 0u)
			{
				const float noiseStd = 0.05f;
				unsigned int rng = noiseSeed ^ (li * 65537u); // per-layer variation
				for (size_t i = 0; i < outVol; i += 2u)
				{
					// Box-Muller with Numerical Recipes LCG (thread-safe, no shared state)
					float u1, u2;
					do {
						rng = rng * 1664525u + 1013904223u;
						u1 = static_cast<float>(rng & 0x7FFFFFFFu) / 2147483648.0f;
					} while (u1 < 1e-10f);
					rng = rng * 1664525u + 1013904223u;
					u2 = static_cast<float>(rng & 0x7FFFFFFFu) / 2147483648.0f;
					const float r = sqrtf(-2.0f * logf(u1));
					const float theta = 6.2831853f * u2;
					scratch[2u + li][i] += noiseStd * r * cosf(theta);
					if (i + 1u < outVol)
						scratch[2u + li][i + 1u] += noiseStd * r * sinf(theta);
				}
			}
		}
		else if (dl.useTanh)
		{
			for (size_t i = 0; i < outVol; ++i)
				scratch[2u + li][i] = tanhf(scratch[2u + li][i]);
		}
		else
		{
			// Temperature-scaled sigmoid: sigmoid(z / T).
			// Higher T widens the linear region, preventing binary saturation.
			// With T=3, reaching output 0.98 requires z≈12 — weight decay resists this.
			const float invTemp = 1.0f / config.genOutputTemp;
			for (size_t i = 0; i < outVol; ++i)
				scratch[2u + li][i] = sigmoid(scratch[2u + li][i] * invTemp);
		}

		curData = &scratch[2u + li][0];
	}

	// Output is the last layer's activation
	output = scratch[numScratch - 1u];

	if (scratchOut)
		*scratchOut = scratch;
}

void GAN::deconvBackward(NNetwork& net, const std::vector<std::vector<float> >& scratch,
                         const float* outputGrad, unsigned int outputSize,
                         std::vector<float>* inputGrad,
                         DeconvScratchArena* arena)
{
	NNetwork::TensorDeconvState& ds = net.tensorDeconv;
	const unsigned int numLayers = static_cast<unsigned int>(ds.layers.size());

	// Scratch layout (from deconvForward):
	// scratch[0] = original input copy
	// scratch[1] = FC output (post-activation)
	// scratch[2..numLayers+1] = deconv layer outputs (post-activation)
	// scratch[numLayers+2..2*numLayers+1] = BN scratch (invstd + x_hat)
	const unsigned int numScratch = 2u + numLayers;

	float* dCurPtr;
	std::vector<float> dCurLocal;
	if (arena && arena->initialized) {
		dCurPtr = &arena->dCur[0];
		std::memcpy(dCurPtr, outputGrad, outputSize * sizeof(float));
	} else {
		dCurLocal.assign(outputGrad, outputGrad + outputSize);
		dCurPtr = &dCurLocal[0];
	}

	// Backward through deconv layers (reverse order)
	for (int li = static_cast<int>(numLayers) - 1; li >= 0; --li)
	{
		const unsigned int l = static_cast<unsigned int>(li);
		NNetwork::TensorDeconvState::DeconvLayer& dl = ds.layers[l];
		const unsigned int outVol = dl.outC * dl.outH * dl.outW;

		// Activation derivative
		const std::vector<float>& layerOut = scratch[2u + l];
		if (dl.useReLU)
		{
			for (size_t i = 0; i < outVol; ++i)
				dCurPtr[i] = (layerOut[i] > 0.0f) ? dCurPtr[i] : 0.0f;
		}
		else if (dl.useTanh)
		{
			// tanh derivative: d/dz tanh(z) = 1 - tanh(z)^2
			for (size_t i = 0; i < outVol; ++i)
			{
				const float t = layerOut[i];
				dCurPtr[i] *= (1.0f - t * t);
			}
		}
		else
		{
			// Temperature-scaled sigmoid derivative: d/dz sigmoid(z/T) = a*(1-a) / T
			const float invTemp = 1.0f / config.genOutputTemp;
			for (size_t i = 0; i < outVol; ++i)
				dCurPtr[i] *= sigmoidDeriv(layerOut[i]) * invTemp;
		}

		// Batch norm backward
		if (dl.useBatchNorm)
		{
			const unsigned int HW = dl.outH * dl.outW;
			const float invHW = 1.0f / static_cast<float>(HW);
			const float* invstdBuf = &scratch[numScratch + l][0];
			const float* xhatBuf = &scratch[numScratch + l][dl.outC];

			for (unsigned int c = 0; c < dl.outC; ++c)
			{
				const size_t chOff = static_cast<size_t>(c) * HW;
				// Gamma/beta gradients
				float dGamma = 0.0f, dBeta = 0.0f;
				for (unsigned int p = 0; p < HW; ++p)
				{
					dGamma += dCurPtr[chOff + p] * xhatBuf[chOff + p];
					dBeta += dCurPtr[chOff + p];
				}
				dl.gBnGamma[c] += dGamma;
				dl.gBnBeta[c] += dBeta;

				// Input gradient through BN
				const float invstd = invstdBuf[c];
				const float gamma = dl.bnGamma[c];
				float sumDyG = 0.0f, sumDyGXh = 0.0f;
				for (unsigned int p = 0; p < HW; ++p)
				{
					const float dyg = dCurPtr[chOff + p] * gamma;
					sumDyG += dyg;
					sumDyGXh += dyg * xhatBuf[chOff + p];
				}
				const float meanDyG = sumDyG * invHW;
				const float meanDyGXh = sumDyGXh * invHW;
				for (unsigned int p = 0; p < HW; ++p)
				{
					const float dyg = dCurPtr[chOff + p] * gamma;
					dCurPtr[chOff + p] = invstd * (dyg - meanDyG - xhatBuf[chOff + p] * meanDyGXh);
				}
			}
		}

		// Bias gradient
		for (unsigned int c = 0; c < dl.outC; ++c)
		{
			float bsum = 0.0f;
			const size_t chOff = static_cast<size_t>(c) * dl.outH * dl.outW;
			for (unsigned int p = 0; p < dl.outH * dl.outW; ++p)
				bsum += dCurPtr[chOff + p];
			dl.gBias[c] += bsum;
		}

		// Layer input is scratch[1+l]: scratch[1] = FC output for l=0, scratch[2..] for l>0
		const float* layerInput = &scratch[1u + l][0];

		if (dl.useUpsampleConv)
		{
			// Standard conv backward (upsample+conv path)
			const unsigned int upH = dl.inH * dl.strideH;
			const unsigned int upW = dl.inW * dl.strideW;
			const unsigned int K = dl.inC * dl.kH * dl.kW;
			const unsigned int N = dl.outH * dl.outW;

			// Recompute forward cols: upsample input, im2col
			float* upsampledPtr;
			std::vector<float> upsampledLocal;
			if (arena && arena->initialized) {
				upsampledPtr = &arena->upsampled[l][0];
			} else {
				upsampledLocal.resize(static_cast<size_t>(dl.inC) * upH * upW);
				upsampledPtr = &upsampledLocal[0];
			}
			nn_upsample_cpu(layerInput, dl.inC, dl.inH, dl.inW, dl.strideH, dl.strideW, upsampledPtr);

			float* colsPtr;
			std::vector<float> colsLocal;
			if (arena && arena->initialized) {
				colsPtr = &arena->cols[l][0];
			} else {
				colsLocal.resize(static_cast<size_t>(N) * K);
				colsPtr = &colsLocal[0];
			}
			im2col_cpu(upsampledPtr, dl.inC, upH, upW,
			           dl.kH, dl.kW, 1u, 1u, dl.padH, dl.padW,
			           dl.outH, dl.outW, colsPtr);

			// Weight gradient: gW[outC, K] += dCur[outC, N] * cols[N, K]
			sgemm_cpu(dCurPtr, colsPtr, &dl.gW[0], dl.outC, N, K);

			// Input gradient: dCols[N, K] = dCur^T[N, outC] * W[outC, K]
			float* dColsPtr;
			std::vector<float> dColsLocal;
			if (arena && arena->initialized) {
				dColsPtr = &arena->dCols[l][0];
				std::memset(dColsPtr, 0, static_cast<size_t>(N) * K * sizeof(float));
			} else {
				dColsLocal.assign(static_cast<size_t>(N) * K, 0.0f);
				dColsPtr = &dColsLocal[0];
			}
			sgemm_atb_cpu(dCurPtr, &dl.W[0], dColsPtr, dl.outC, N, K);

			// col2im to get d_upsampled[inC, upH, upW]
			float* dUpPtr;
			std::vector<float> dUpLocal;
			if (arena && arena->initialized) {
				dUpPtr = &arena->dUp[l][0];
				std::memset(dUpPtr, 0, static_cast<size_t>(dl.inC) * upH * upW * sizeof(float));
			} else {
				dUpLocal.assign(static_cast<size_t>(dl.inC) * upH * upW, 0.0f);
				dUpPtr = &dUpLocal[0];
			}
			col2im_cpu(dColsPtr, dl.inC, upH, upW,
			           dl.kH, dl.kW, 1u, 1u, dl.padH, dl.padW,
			           dl.outH, dl.outW, dUpPtr);

			// Downsample to get dInput[inC, inH, inW]
			const size_t inputVol = static_cast<size_t>(dl.inC) * dl.inH * dl.inW;
			float* dInputPtr;
			std::vector<float> dInputLocal;
			if (arena && arena->initialized) {
				dInputPtr = &arena->dInput[l][0];
			} else {
				dInputLocal.resize(inputVol);
				dInputPtr = &dInputLocal[0];
			}
			nn_downsample_sum_cpu(dUpPtr, dl.inC, upH, upW, dl.strideH, dl.strideW, dInputPtr);
			if (arena && arena->initialized) {
				std::memcpy(dCurPtr, &arena->dInput[l][0], inputVol * sizeof(float));
			} else {
				dCurLocal = dInputLocal;
				dCurPtr = &dCurLocal[0];
			}
		}
		else
		{
			// Transposed conv backward
			const unsigned int N = dl.inH * dl.inW;
			const unsigned int outK = dl.outC * dl.kH * dl.kW;

			// im2col on dCur: gradient of col2im is im2col
			float* dOutputColsPtr;
			std::vector<float> dOutputColsLocal;
			if (arena && arena->initialized) {
				dOutputColsPtr = &arena->dCols[l][0];
				std::memset(dOutputColsPtr, 0, static_cast<size_t>(outK) * N * sizeof(float));
			} else {
				dOutputColsLocal.assign(static_cast<size_t>(outK) * N, 0.0f);
				dOutputColsPtr = &dOutputColsLocal[0];
			}
			im2col_cpu(dCurPtr, dl.outC, dl.outH, dl.outW,
			           dl.kH, dl.kW, dl.strideH, dl.strideW, dl.padH, dl.padW,
			           dl.inH, dl.inW, dOutputColsPtr);

			// Weight gradient: dW[inC, outK] += input_flat[inC, N] * dOutputCols[N, outK]^T
			sgemm_abt_cpu(layerInput, dOutputColsPtr, &dl.gW[0], dl.inC, N, outK);

			// Input gradient: dinput[inC, N] = W[inC, outK] * dOutputCols[outK, N]
			{
				const size_t inputVol = static_cast<size_t>(dl.inC) * N;
				float* dInputPtr;
				std::vector<float> dInputLocal;
				if (arena && arena->initialized) {
					dInputPtr = &arena->dInput[l][0];
					std::memset(dInputPtr, 0, inputVol * sizeof(float));
				} else {
					dInputLocal.assign(inputVol, 0.0f);
					dInputPtr = &dInputLocal[0];
				}
				sgemm_cpu(&dl.W[0], dOutputColsPtr, dInputPtr, dl.inC, outK, N);
				if (arena && arena->initialized) {
					std::memcpy(dCurPtr, &arena->dInput[l][0], inputVol * sizeof(float));
				} else {
					dCurLocal = dInputLocal;
					dCurPtr = &dCurLocal[0];
				}
			}
		}
	}

	// FC backward: dCur now holds gradient w.r.t. FC output
	// scratch[1] = FC output (post-activation)
	float* dFCBuf;
	std::vector<float> dFCLocal;
	if (arena && arena->initialized) {
		dFCBuf = &arena->dFC[0];
		std::memset(dFCBuf, 0, ds.fcOut * sizeof(float));
	} else {
		dFCLocal.assign(ds.fcOut, 0.0f);
		dFCBuf = &dFCLocal[0];
	}
	for (unsigned int j = 0; j < ds.fcOut; ++j)
	{
		const float a = scratch[1][j];
		dFCBuf[j] = dCurPtr[j] * ((a > 0.0f) ? 1.0f : 0.2f); // LeakyReLU derivative
	}

	// FC bias gradient
	for (unsigned int j = 0; j < ds.fcOut; ++j)
		ds.fcGBias[j] += dFCBuf[j];

	// FC weight gradient: gW[j*fcIn + i] += dFC[j] * input[i]
	const std::vector<float>& inputCopy = scratch[0];
	for (unsigned int j = 0; j < ds.fcOut; ++j)
	{
		const float d = dFCBuf[j];
		const size_t rowOff = static_cast<size_t>(j) * ds.fcIn;
		for (unsigned int i = 0; i < ds.fcIn; ++i)
			ds.fcGW[rowOff + i] += d * inputCopy[i];
	}

	// Input gradient (if requested)
	if (inputGrad)
	{
		inputGrad->assign(ds.fcIn, 0.0f);
		for (unsigned int i = 0; i < ds.fcIn; ++i)
		{
			float sum = 0.0f;
			for (unsigned int j = 0; j < ds.fcOut; ++j)
				sum += dFCBuf[j] * ds.fcW[static_cast<size_t>(j) * ds.fcIn + i];
			(*inputGrad)[i] = sum;
		}
	}
}

// ============================================================
// Deconv Backward Pass (GradientBuffer overload - thread-safe)
// ============================================================

void GAN::deconvBackward(const NNetwork& net, const std::vector<std::vector<float> >& scratch,
                         const float* outputGrad, unsigned int outputSize,
                         std::vector<float>* inputGrad,
                         GradientBuffer& gradBuf,
                         DeconvScratchArena* arena)
{
	const NNetwork::TensorDeconvState& ds = net.tensorDeconv;
	const unsigned int numLayers = static_cast<unsigned int>(ds.layers.size());

	// Scratch layout (from deconvForward):
	// scratch[0] = original input copy
	// scratch[1] = FC output (post-activation)
	// scratch[2..numLayers+1] = deconv layer outputs (post-activation)
	// scratch[numLayers+2..2*numLayers+1] = BN scratch (invstd + x_hat)
	const unsigned int numScratch = 2u + numLayers;

	float* dCurPtr;
	std::vector<float> dCurLocal;
	if (arena && arena->initialized) {
		dCurPtr = &arena->dCur[0];
		std::memcpy(dCurPtr, outputGrad, outputSize * sizeof(float));
	} else {
		dCurLocal.assign(outputGrad, outputGrad + outputSize);
		dCurPtr = &dCurLocal[0];
	}

	// Backward through deconv layers (reverse order)
	for (int li = static_cast<int>(numLayers) - 1; li >= 0; --li)
	{
		const unsigned int l = static_cast<unsigned int>(li);
		const NNetwork::TensorDeconvState::DeconvLayer& dl = ds.layers[l];
		const unsigned int outVol = dl.outC * dl.outH * dl.outW;

		// Activation derivative
		const std::vector<float>& layerOut = scratch[2u + l];
		if (dl.useReLU)
		{
			for (size_t i = 0; i < outVol; ++i)
				dCurPtr[i] = (layerOut[i] > 0.0f) ? dCurPtr[i] : 0.0f;
		}
		else if (dl.useTanh)
		{
			// tanh derivative: d/dz tanh(z) = 1 - tanh(z)^2
			for (size_t i = 0; i < outVol; ++i)
			{
				const float t = layerOut[i];
				dCurPtr[i] *= (1.0f - t * t);
			}
		}
		else
		{
			// Temperature-scaled sigmoid derivative: d/dz sigmoid(z/T) = a*(1-a) / T
			const float invTemp = 1.0f / config.genOutputTemp;
			for (size_t i = 0; i < outVol; ++i)
				dCurPtr[i] *= sigmoidDeriv(layerOut[i]) * invTemp;
		}

		// Batch norm backward
		if (dl.useBatchNorm)
		{
			const unsigned int HW = dl.outH * dl.outW;
			const float invHW = 1.0f / static_cast<float>(HW);
			const float* invstdBuf = &scratch[numScratch + l][0];
			const float* xhatBuf = &scratch[numScratch + l][dl.outC];

			for (unsigned int c = 0; c < dl.outC; ++c)
			{
				const size_t chOff = static_cast<size_t>(c) * HW;
				float dGamma = 0.0f, dBeta = 0.0f;
				for (unsigned int p = 0; p < HW; ++p)
				{
					dGamma += dCurPtr[chOff + p] * xhatBuf[chOff + p];
					dBeta += dCurPtr[chOff + p];
				}
				gradBuf.deconvGBnGamma[l][c] += dGamma;
				gradBuf.deconvGBnBeta[l][c] += dBeta;

				const float invstd = invstdBuf[c];
				const float gamma = dl.bnGamma[c];
				float sumDyG = 0.0f, sumDyGXh = 0.0f;
				for (unsigned int p = 0; p < HW; ++p)
				{
					const float dyg = dCurPtr[chOff + p] * gamma;
					sumDyG += dyg;
					sumDyGXh += dyg * xhatBuf[chOff + p];
				}
				const float meanDyG = sumDyG * invHW;
				const float meanDyGXh = sumDyGXh * invHW;
				for (unsigned int p = 0; p < HW; ++p)
				{
					const float dyg = dCurPtr[chOff + p] * gamma;
					dCurPtr[chOff + p] = invstd * (dyg - meanDyG - xhatBuf[chOff + p] * meanDyGXh);
				}
			}
		}

		// Bias gradient into GradientBuffer
		for (unsigned int c = 0; c < dl.outC; ++c)
		{
			float bsum = 0.0f;
			const size_t chOff = static_cast<size_t>(c) * dl.outH * dl.outW;
			for (unsigned int p = 0; p < dl.outH * dl.outW; ++p)
				bsum += dCurPtr[chOff + p];
			gradBuf.deconvGBias[l][c] += bsum;
		}

		// Layer input is scratch[1+l]
		const float* layerInput = &scratch[1u + l][0];

		if (dl.useUpsampleConv)
		{
			// Standard conv backward (upsample+conv path)
			const unsigned int upH = dl.inH * dl.strideH;
			const unsigned int upW = dl.inW * dl.strideW;
			const unsigned int K = dl.inC * dl.kH * dl.kW;
			const unsigned int N = dl.outH * dl.outW;

			// Recompute forward cols: upsample input, im2col
			float* upsampledPtr;
			std::vector<float> upsampledLocal;
			if (arena && arena->initialized) {
				upsampledPtr = &arena->upsampled[l][0];
			} else {
				upsampledLocal.resize(static_cast<size_t>(dl.inC) * upH * upW);
				upsampledPtr = &upsampledLocal[0];
			}
			nn_upsample_cpu(layerInput, dl.inC, dl.inH, dl.inW, dl.strideH, dl.strideW, upsampledPtr);

			float* colsPtr;
			std::vector<float> colsLocal;
			if (arena && arena->initialized) {
				colsPtr = &arena->cols[l][0];
			} else {
				colsLocal.resize(static_cast<size_t>(N) * K);
				colsPtr = &colsLocal[0];
			}
			im2col_cpu(upsampledPtr, dl.inC, upH, upW,
			           dl.kH, dl.kW, 1u, 1u, dl.padH, dl.padW,
			           dl.outH, dl.outW, colsPtr);

			// Weight gradient: gW[outC, K] += dCur[outC, N] * cols[N, K]
			sgemm_cpu(dCurPtr, colsPtr, &gradBuf.deconvGW[l][0], dl.outC, N, K);

			// Input gradient: dCols[N, K] = dCur^T[N, outC] * W[outC, K]
			float* dColsPtr;
			std::vector<float> dColsLocal;
			if (arena && arena->initialized) {
				dColsPtr = &arena->dCols[l][0];
				std::memset(dColsPtr, 0, static_cast<size_t>(N) * K * sizeof(float));
			} else {
				dColsLocal.assign(static_cast<size_t>(N) * K, 0.0f);
				dColsPtr = &dColsLocal[0];
			}
			sgemm_atb_cpu(dCurPtr, &dl.W[0], dColsPtr, dl.outC, N, K);

			// col2im to get d_upsampled[inC, upH, upW]
			float* dUpPtr;
			std::vector<float> dUpLocal;
			if (arena && arena->initialized) {
				dUpPtr = &arena->dUp[l][0];
				std::memset(dUpPtr, 0, static_cast<size_t>(dl.inC) * upH * upW * sizeof(float));
			} else {
				dUpLocal.assign(static_cast<size_t>(dl.inC) * upH * upW, 0.0f);
				dUpPtr = &dUpLocal[0];
			}
			col2im_cpu(dColsPtr, dl.inC, upH, upW,
			           dl.kH, dl.kW, 1u, 1u, dl.padH, dl.padW,
			           dl.outH, dl.outW, dUpPtr);

			// Downsample to get dInput[inC, inH, inW]
			const size_t inputVol = static_cast<size_t>(dl.inC) * dl.inH * dl.inW;
			float* dInputPtr;
			std::vector<float> dInputLocal;
			if (arena && arena->initialized) {
				dInputPtr = &arena->dInput[l][0];
			} else {
				dInputLocal.resize(inputVol);
				dInputPtr = &dInputLocal[0];
			}
			nn_downsample_sum_cpu(dUpPtr, dl.inC, upH, upW, dl.strideH, dl.strideW, dInputPtr);
			if (arena && arena->initialized) {
				std::memcpy(dCurPtr, &arena->dInput[l][0], inputVol * sizeof(float));
			} else {
				dCurLocal = dInputLocal;
				dCurPtr = &dCurLocal[0];
			}
		}
		else
		{
			// Transposed conv backward
			const unsigned int N = dl.inH * dl.inW;
			const unsigned int outK = dl.outC * dl.kH * dl.kW;

			// im2col on dCur: gradient of col2im is im2col
			float* dOutputColsPtr;
			std::vector<float> dOutputColsLocal;
			if (arena && arena->initialized) {
				dOutputColsPtr = &arena->dCols[l][0];
				std::memset(dOutputColsPtr, 0, static_cast<size_t>(outK) * N * sizeof(float));
			} else {
				dOutputColsLocal.assign(static_cast<size_t>(outK) * N, 0.0f);
				dOutputColsPtr = &dOutputColsLocal[0];
			}
			im2col_cpu(dCurPtr, dl.outC, dl.outH, dl.outW,
			           dl.kH, dl.kW, dl.strideH, dl.strideW, dl.padH, dl.padW,
			           dl.inH, dl.inW, dOutputColsPtr);

			// Weight gradient into GradientBuffer
			sgemm_abt_cpu(layerInput, dOutputColsPtr, &gradBuf.deconvGW[l][0], dl.inC, N, outK);

			// Input gradient: dinput[inC, N] = W[inC, outK] * dOutputCols[outK, N]
			{
				const size_t inputVol = static_cast<size_t>(dl.inC) * N;
				float* dInputPtr;
				std::vector<float> dInputLocal;
				if (arena && arena->initialized) {
					dInputPtr = &arena->dInput[l][0];
					std::memset(dInputPtr, 0, inputVol * sizeof(float));
				} else {
					dInputLocal.assign(inputVol, 0.0f);
					dInputPtr = &dInputLocal[0];
				}
				sgemm_cpu(&dl.W[0], dOutputColsPtr, dInputPtr, dl.inC, outK, N);
				if (arena && arena->initialized) {
					std::memcpy(dCurPtr, &arena->dInput[l][0], inputVol * sizeof(float));
				} else {
					dCurLocal = dInputLocal;
					dCurPtr = &dCurLocal[0];
				}
			}
		}
	}

	// FC backward: dCur now holds gradient w.r.t. FC output
	// scratch[1] = FC output (post-activation)
	float* dFCBuf;
	std::vector<float> dFCLocal;
	if (arena && arena->initialized) {
		dFCBuf = &arena->dFC[0];
		std::memset(dFCBuf, 0, ds.fcOut * sizeof(float));
	} else {
		dFCLocal.assign(ds.fcOut, 0.0f);
		dFCBuf = &dFCLocal[0];
	}
	for (unsigned int j = 0; j < ds.fcOut; ++j)
	{
		const float a = scratch[1][j];
		dFCBuf[j] = dCurPtr[j] * ((a > 0.0f) ? 1.0f : 0.2f); // LeakyReLU derivative
	}

	// FC bias gradient into GradientBuffer
	for (unsigned int j = 0; j < ds.fcOut; ++j)
		gradBuf.deconvFcGBias[j] += dFCBuf[j];

	// FC weight gradient into GradientBuffer
	const std::vector<float>& inputCopy = scratch[0];
	for (unsigned int j = 0; j < ds.fcOut; ++j)
	{
		const float d = dFCBuf[j];
		const size_t rowOff = static_cast<size_t>(j) * ds.fcIn;
		for (unsigned int i = 0; i < ds.fcIn; ++i)
			gradBuf.deconvFcGW[rowOff + i] += d * inputCopy[i];
	}

	// Input gradient (if requested)
	if (inputGrad)
	{
		inputGrad->assign(ds.fcIn, 0.0f);
		for (unsigned int i = 0; i < ds.fcIn; ++i)
		{
			float sum = 0.0f;
			for (unsigned int j = 0; j < ds.fcOut; ++j)
				sum += dFCBuf[j] * ds.fcW[static_cast<size_t>(j) * ds.fcIn + i];
			(*inputGrad)[i] = sum;
		}
	}
}

void GAN::deconvUpdate(NNetwork& net, float lr)
{
	NNetwork::TensorDeconvState& ds = net.tensorDeconv;
	++ds.optimizerStep;
	const float beta1 = config.adamBeta1;
	const float beta2 = config.adamBeta2;
	const float eps = config.adamEps;
	const float bc1 = 1.0f - powf(beta1, static_cast<float>(ds.optimizerStep));
	const float bc2 = 1.0f - powf(beta2, static_cast<float>(ds.optimizerStep));

	const float wd = config.weightDecay;

	// FC projection
	for (size_t i = 0; i < ds.fcW.size(); ++i)
	{
		const float g = ds.fcGW[i];
		ds.fcVW[i] = beta1 * ds.fcVW[i] + (1.0f - beta1) * g;
		ds.fcV2W[i] = beta2 * ds.fcV2W[i] + (1.0f - beta2) * g * g;
		const float mHat = ds.fcVW[i] / bc1;
		const float vHat = ds.fcV2W[i] / bc2;
		ds.fcW[i] -= lr * (mHat / (sqrtf(vHat) + eps) + wd * ds.fcW[i]);
		ds.fcGW[i] = 0.0f;
	}
	for (unsigned int j = 0; j < ds.fcOut; ++j)
	{
		const float g = ds.fcGBias[j];
		ds.fcVBias[j] = beta1 * ds.fcVBias[j] + (1.0f - beta1) * g;
		ds.fcV2Bias[j] = beta2 * ds.fcV2Bias[j] + (1.0f - beta2) * g * g;
		const float mHat = ds.fcVBias[j] / bc1;
		const float vHat = ds.fcV2Bias[j] / bc2;
		ds.fcBias[j] -= lr * mHat / (sqrtf(vHat) + eps);
		ds.fcGBias[j] = 0.0f;
	}

	// Deconv layers
	for (unsigned int l = 0; l < ds.layers.size(); ++l)
	{
		NNetwork::TensorDeconvState::DeconvLayer& dl = ds.layers[l];
		for (size_t i = 0; i < dl.W.size(); ++i)
		{
			const float g = dl.gW[i];
			dl.vW[i] = beta1 * dl.vW[i] + (1.0f - beta1) * g;
			dl.v2W[i] = beta2 * dl.v2W[i] + (1.0f - beta2) * g * g;
			const float mHat = dl.vW[i] / bc1;
			const float vHat = dl.v2W[i] / bc2;
			dl.W[i] -= lr * (mHat / (sqrtf(vHat) + eps) + wd * dl.W[i]);
			dl.gW[i] = 0.0f;
		}
		for (unsigned int c = 0; c < dl.outC; ++c)
		{
			const float g = dl.gBias[c];
			dl.vBias[c] = beta1 * dl.vBias[c] + (1.0f - beta1) * g;
			dl.v2Bias[c] = beta2 * dl.v2Bias[c] + (1.0f - beta2) * g * g;
			const float mHat = dl.vBias[c] / bc1;
			const float vHat = dl.v2Bias[c] / bc2;
			dl.bias[c] -= lr * mHat / (sqrtf(vHat) + eps);
			dl.gBias[c] = 0.0f;
		}
		if (dl.useBatchNorm)
		{
			for (unsigned int c = 0; c < dl.outC; ++c)
			{
				{
					const float g = dl.gBnGamma[c];
					dl.vBnGamma[c] = beta1 * dl.vBnGamma[c] + (1.0f - beta1) * g;
					dl.v2BnGamma[c] = beta2 * dl.v2BnGamma[c] + (1.0f - beta2) * g * g;
					const float mHat = dl.vBnGamma[c] / bc1;
					const float vHat = dl.v2BnGamma[c] / bc2;
					dl.bnGamma[c] -= lr * mHat / (sqrtf(vHat) + eps);
					dl.gBnGamma[c] = 0.0f;
				}
				{
					const float g = dl.gBnBeta[c];
					dl.vBnBeta[c] = beta1 * dl.vBnBeta[c] + (1.0f - beta1) * g;
					dl.v2BnBeta[c] = beta2 * dl.v2BnBeta[c] + (1.0f - beta2) * g * g;
					const float mHat = dl.vBnBeta[c] / bc1;
					const float vHat = dl.v2BnBeta[c] / bc2;
					dl.bnBeta[c] -= lr * mHat / (sqrtf(vHat) + eps);
					dl.gBnBeta[c] = 0.0f;
				}
			}
		}
	}
}

void GAN::zeroDeconvGrads(NNetwork& net)
{
	NNetwork::TensorDeconvState& ds = net.tensorDeconv;
	if (!ds.initialized)
		return;
	std::memset(&ds.fcGW[0], 0, ds.fcGW.size() * sizeof(float));
	std::memset(&ds.fcGBias[0], 0, ds.fcGBias.size() * sizeof(float));
	for (unsigned int l = 0; l < ds.layers.size(); ++l)
	{
		NNetwork::TensorDeconvState::DeconvLayer& dl = ds.layers[l];
		std::memset(&dl.gW[0], 0, dl.gW.size() * sizeof(float));
		std::memset(&dl.gBias[0], 0, dl.gBias.size() * sizeof(float));
		if (dl.useBatchNorm)
		{
			std::memset(&dl.gBnGamma[0], 0, dl.gBnGamma.size() * sizeof(float));
			std::memset(&dl.gBnBeta[0], 0, dl.gBnBeta.size() * sizeof(float));
		}
	}
}

// ============================================================
// Loss functions
// ============================================================

float GAN::vanillaDiscriminatorLoss(float predReal, float predFake) const
{
	// -[log(D(x)) + log(1-D(G(z)))]
	const float eps = 1e-7f;
	const float pr = (predReal < eps) ? eps : ((predReal > 1.0f - eps) ? 1.0f - eps : predReal);
	const float pf = (predFake < eps) ? eps : ((predFake > 1.0f - eps) ? 1.0f - eps : predFake);
	return -(logf(pr) + logf(1.0f - pf));
}

float GAN::vanillaGeneratorLoss(float predFake) const
{
	// -log(D(G(z)))  [non-saturating]
	const float eps = 1e-7f;
	const float pf = (predFake < eps) ? eps : ((predFake > 1.0f - eps) ? 1.0f - eps : predFake);
	return -logf(pf);
}

float GAN::wganDiscriminatorLoss(float predReal, float predFake) const
{
	// D(G(z)) - D(x)  (critic wants to maximize D(x) - D(G(z)))
	return predFake - predReal;
}

float GAN::wganGeneratorLoss(float predFake) const
{
	// -D(G(z))
	return -predFake;
}

float GAN::lsganDiscriminatorLoss(float predReal, float predFake) const
{
	// 0.5 * [(D(x) - 1)^2 + D(G(z))^2]
	return 0.5f * ((predReal - 1.0f) * (predReal - 1.0f) + predFake * predFake);
}

float GAN::lsganGeneratorLoss(float predFake) const
{
	// 0.5 * (D(G(z)) - 1)^2
	return 0.5f * (predFake - 1.0f) * (predFake - 1.0f);
}

// ============================================================
// WGAN-GP Gradient Penalty
// ============================================================

float GAN::computeGradientPenalty(const float* real, const float* fake,
                                  unsigned int dim)
{
	return computeGradientPenalty(discriminator, real, fake, dim);
}

float GAN::computeGradientPenalty(NNetwork& disc, const float* real, const float* fake,
                                  unsigned int dim)
{
	// Interpolate: x_hat = eps * real + (1-eps) * fake
	const float epsilon = glades::rng::unit_float01(rngEngine);

	std::vector<float> xHat(dim);
	for (unsigned int i = 0; i < dim; ++i)
		xHat[i] = epsilon * real[i] + (1.0f - epsilon) * fake[i];

	// Forward discriminator on interpolated sample
	std::vector<float> discOutput;
	if (config.archType == GANConfig::GAN_DFF)
	{
		std::vector<std::vector<float> > discAct;
		dffForward(disc, &xHat[0], dim, discAct, false);

		// Backward to get input gradient
		float one = 1.0f;
		std::vector<float> dInputHat;
		dffBackward(disc, discAct, &one, 1u, &dInputHat, false);

		// Zero out the accumulated weight gradients (we only wanted input grad)
		for (unsigned int t = 0; t < disc.tensorDff.T.size(); ++t)
		{
			NNetwork::TensorDFFState::Transition& tr = disc.tensorDff.T[t];
			std::memset(&tr.gW[0], 0, tr.gW.size() * sizeof(float));
			std::memset(&tr.gBias[0], 0, tr.gBias.size() * sizeof(float));
		}

		// Compute ||grad||_2
		float gradNormSq = 0.0f;
		for (unsigned int i = 0; i < dim; ++i)
			gradNormSq += dInputHat[i] * dInputHat[i];
		const float gradNorm = sqrtf(gradNormSq);
		const float penalty = (gradNorm - 1.0f) * (gradNorm - 1.0f);
		return penalty;
	}
	else
	{
		// CNN path
		std::vector<float> cnnOut;
		cnnForward(disc, &xHat[0], cnnOut, false);

		float one = 1.0f;
		std::vector<float> dInputHat;
		cnnBackward(disc, &xHat[0], &one, 1u, &dInputHat, NULL, false);

		// Zero out accumulated weight gradients
		for (unsigned int l = 0; l < disc.tensorCnn.convLayers.size(); ++l)
		{
			NNetwork::TensorCNNState::ConvLayer& cl = disc.tensorCnn.convLayers[l];
			std::memset(&cl.gW[0], 0, cl.gW.size() * sizeof(float));
			std::memset(&cl.gBias[0], 0, cl.gBias.size() * sizeof(float));
		}
		for (unsigned int t = 0; t < disc.tensorCnn.fcLayers.size(); ++t)
		{
			NNetwork::TensorCNNState::FCTransition& fc = disc.tensorCnn.fcLayers[t];
			std::memset(&fc.gW[0], 0, fc.gW.size() * sizeof(float));
			std::memset(&fc.gBias[0], 0, fc.gBias.size() * sizeof(float));
		}

		float gradNormSq = 0.0f;
		for (unsigned int i = 0; i < dim; ++i)
			gradNormSq += dInputHat[i] * dInputHat[i];
		const float gradNorm = sqrtf(gradNormSq);
		return (gradNorm - 1.0f) * (gradNorm - 1.0f);
	}
}

// ============================================================
// WGAN-GP Gradient Penalty (GradientBuffer overload - thread-safe)
// ============================================================

float GAN::computeGradientPenalty(const NNetwork& disc, const float* real, const float* fake,
                                  unsigned int dim, float epsilon,
                                  GradientBuffer& discardBuf,
                                  std::vector<std::vector<float> >& scratchDeltaLocal)
{
	// Interpolate: x_hat = eps * real + (1-eps) * fake
	std::vector<float> xHat(dim);
	for (unsigned int i = 0; i < dim; ++i)
		xHat[i] = epsilon * real[i] + (1.0f - epsilon) * fake[i];

	// Forward discriminator on interpolated sample
	if (config.archType == GANConfig::GAN_DFF)
	{
		std::vector<std::vector<float> > discAct;
		dffForward(disc, &xHat[0], dim, discAct, false);

		// Backward to get input gradient - grads go to discardBuf
		float one = 1.0f;
		std::vector<float> dInputHat;
		dffBackward(disc, discAct, &one, 1u, &dInputHat,
		            discardBuf, scratchDeltaLocal, false);

		// No need to zero network grads - they went to discardBuf

		// Compute ||grad||_2
		float gradNormSq = 0.0f;
		for (unsigned int i = 0; i < dim; ++i)
			gradNormSq += dInputHat[i] * dInputHat[i];
		const float gradNorm = sqrtf(gradNormSq);
		const float penalty = (gradNorm - 1.0f) * (gradNorm - 1.0f);
		return penalty;
	}
	else
	{
		// CNN path
		std::vector<float> cnnOut;
		cnnForward(disc, &xHat[0], cnnOut, false);

		float one = 1.0f;
		std::vector<float> dInputHat;
		cnnBackward(disc, &xHat[0], &one, 1u, &dInputHat, discardBuf, NULL, false);

		// No need to zero network grads - they went to discardBuf

		float gradNormSq = 0.0f;
		for (unsigned int i = 0; i < dim; ++i)
			gradNormSq += dInputHat[i] * dInputHat[i];
		const float gradNorm = sqrtf(gradNormSq);
		return (gradNorm - 1.0f) * (gradNorm - 1.0f);
	}
}

// ============================================================
// Main Training Loop - dispatch
// ============================================================

NNetworkStatus GAN::train(const DataInput* realData, IGANCallbacks* cb)
{
	if (!realData)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "GAN::train: realData is NULL");

	const bool hasCycle = config.useCycle || (config.variantType == GANConfig::GAN_CYCLE);
	if (hasCycle)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		    "GAN::train: CycleGAN requires two-dataset train(domainA, domainB)");

	return trainSingleDomain(realData, cb);
}

NNetworkStatus GAN::train(const DataInput* domainA, const DataInput* domainB, IGANCallbacks* cb)
{
	if (!domainA || !domainB)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT, "GAN::train: domain data is NULL");

	const bool hasCycle = config.useCycle || (config.variantType == GANConfig::GAN_CYCLE);
	if (!hasCycle)
		return NNetworkStatus(NNetworkStatus::INVALID_ARGUMENT,
		    "GAN::train: two-dataset train is only for CycleGAN");
	return trainDualDomain(domainA, domainB, cb);
}

// ============================================================
// Parallel layer-wise gradient reduction
// ============================================================

void GAN::layerReduceBody(void* userData, unsigned int begin, unsigned int end)
{
	LayerReduceData& d = *static_cast<LayerReduceData*>(userData);

	for (unsigned int layerIdx = begin; layerIdx < end; ++layerIdx)
	{
		if (d.isGen)
		{
			// Deconv generator reduction
			NNetwork::TensorDeconvState& dc = d.net->tensorDeconv;
			if (layerIdx == 0u)
			{
				// FC layer
				std::memset(&dc.fcGW[0], 0, dc.fcGW.size() * sizeof(float));
				std::memset(&dc.fcGBias[0], 0, dc.fcOut * sizeof(float));
				for (unsigned int tid = 0; tid < d.nThreads; ++tid)
				{
					const GradientBuffer& g = d.ctxs[tid].genGrads;
					for (size_t i = 0; i < dc.fcGW.size(); ++i)
						dc.fcGW[i] += g.deconvFcGW[i];
					for (size_t i = 0; i < dc.fcOut; ++i)
						dc.fcGBias[i] += g.deconvFcGBias[i];
				}
			}
			else
			{
				const unsigned int l = layerIdx - 1u;
				NNetwork::TensorDeconvState::DeconvLayer& dl = dc.layers[l];
				std::memset(&dl.gW[0], 0, dl.gW.size() * sizeof(float));
				std::memset(&dl.gBias[0], 0, dl.outC * sizeof(float));
				if (dl.useBatchNorm)
				{
					std::memset(&dl.gBnGamma[0], 0, dl.outC * sizeof(float));
					std::memset(&dl.gBnBeta[0], 0, dl.outC * sizeof(float));
				}
				for (unsigned int tid = 0; tid < d.nThreads; ++tid)
				{
					const GradientBuffer& g = d.ctxs[tid].genGrads;
					for (size_t i = 0; i < dl.gW.size(); ++i)
						dl.gW[i] += g.deconvGW[l][i];
					for (size_t i = 0; i < dl.outC; ++i)
						dl.gBias[i] += g.deconvGBias[l][i];
					if (dl.useBatchNorm)
					{
						for (size_t i = 0; i < dl.outC; ++i)
							dl.gBnGamma[i] += g.deconvGBnGamma[l][i];
						for (size_t i = 0; i < dl.outC; ++i)
							dl.gBnBeta[i] += g.deconvGBnBeta[l][i];
					}
				}
			}
		}
		else
		{
			// CNN discriminator reduction
			NNetwork::TensorCNNState& cnn = d.net->tensorCnn;
			const unsigned int nConv = static_cast<unsigned int>(cnn.convLayers.size());
			if (layerIdx < nConv)
			{
				NNetwork::TensorCNNState::ConvLayer& cl = cnn.convLayers[layerIdx];
				std::memset(&cl.gW[0], 0, cl.gW.size() * sizeof(float));
				std::memset(&cl.gBias[0], 0, cl.outC * sizeof(float));
				for (unsigned int tid = 0; tid < d.nThreads; ++tid)
				{
					const GradientBuffer& g = d.ctxs[tid].discGrads;
					for (size_t i = 0; i < cl.gW.size(); ++i)
						cl.gW[i] += g.convGW[layerIdx][i];
					for (size_t i = 0; i < cl.outC; ++i)
						cl.gBias[i] += g.convGBias[layerIdx][i];
				}
			}
			else
			{
				const unsigned int fcIdx = layerIdx - nConv;
				NNetwork::TensorCNNState::FCTransition& fc = cnn.fcLayers[fcIdx];
				std::memset(&fc.gW[0], 0, fc.gW.size() * sizeof(float));
				std::memset(&fc.gBias[0], 0, fc.out * sizeof(float));
				for (unsigned int tid = 0; tid < d.nThreads; ++tid)
				{
					const GradientBuffer& g = d.ctxs[tid].discGrads;
					for (size_t i = 0; i < fc.gW.size(); ++i)
						fc.gW[i] += g.fcGW[fcIdx][i];
					for (size_t i = 0; i < fc.out; ++i)
						fc.gBias[i] += g.fcGBias[fcIdx][i];
				}
			}
		}
	}
}

// ============================================================
// Parallel discriminator critic callback
// ============================================================

void GAN::discCritBody(void* userData, unsigned int begin, unsigned int end)
{
	DiscCritData& d = *static_cast<DiscCritData*>(userData);
	const unsigned int tid = d.beginToTid[begin];
	GANThreadCtx& ctx = d.ctxs[tid];
	GAN& self = *d.self;

	for (unsigned int s = begin; s < end; ++s)
	{
		const float* realRow = d.realPtrs[s];

		// Build genInput from pre-generated noise + codes
		std::memcpy(&ctx.genInput[0], &d.noise[s][0], self.config.noiseDim * sizeof(float));
		if (d.numCat > 0u)
		{
			for (unsigned int c = 0; c < d.numCat; ++c)
				ctx.genInput[self.config.noiseDim + c] = d.catCodes[s][c] * d.catScale;
		}
		if (d.numCont > 0u)
			std::memcpy(&ctx.genInput[self.config.noiseDim + d.numCat], &d.contCodes[s][0], d.numCont * sizeof(float));

		// Generator forward
		if (d.genIsDeconv)
		{
			self.deconvForward(self.generator, &ctx.genInput[0], d.genInputDim, ctx.deconvOut, &ctx.deconvScratch, tid * 1000003u + s * 7u + 1u,
			                   &ctx.deconvArena);
		}
		else if (d.hasStyle)
		{
			self.dffForward(self.mappingNet, &d.noise[s][0], self.config.noiseDim, ctx.mapAct);
			const std::vector<float>& w = ctx.mapAct.back();
			self.dffForwardStyled(self.generator, &ctx.genInput[0], d.genInputDim, w,
			                      self.styleAffines, self.noiseScales,
			                      ctx.genAct, ctx.sXNorms, ctx.sMeans, ctx.sInvStds, ctx.sNoiseVecs);
		}
		else
		{
			self.dffForward(self.generator, &ctx.genInput[0], d.genInputDim, ctx.genAct,
			                self.config.generatorOutputSigmoid, self.genLN());
		}
		const std::vector<float>& fake = d.genIsDeconv ? ctx.deconvOut : ctx.genAct.back();

		// Discriminator on real
		const bool discSig = (self.config.lossType == GANConfig::GAN_VANILLA);
		float predReal = 0.0f;
		if (self.config.archType == GANConfig::GAN_DFF)
		{
			self.dffForward(self.discriminator, realRow, d.dataDim, ctx.discActReal, discSig);
			predReal = ctx.discActReal.back()[0];
		}
		else
		{
			self.cnnForward(self.discriminator, realRow, ctx.cnnOutReal, discSig);
			predReal = ctx.cnnOutReal[0];
		}

		// Discriminator on fake
		float predFake = 0.0f;
		if (self.config.archType == GANConfig::GAN_DFF)
		{
			self.dffForward(self.discriminator, &fake[0], d.dataDim, ctx.discActFake, discSig);
			predFake = ctx.discActFake.back()[0];
		}
		else
		{
			if (d.hasInfo)
				self.cnnForward(self.discriminator, &fake[0], ctx.cnnOutFake, discSig, &ctx.cnnFcActFake);
			else
				self.cnnForward(self.discriminator, &fake[0], ctx.cnnOutFake, discSig);
			predFake = ctx.cnnOutFake[0];
		}

		// Track raw discriminator outputs for health monitoring
		ctx.dOutRealSum += predReal;
		ctx.dOutFakeSum += predFake;

		// Adversarial loss + backward (using GradientBuffer overloads)
		if (self.config.lossType == GANConfig::GAN_VANILLA)
		{
			// predReal/predFake are already sigmoid outputs from forward
			ctx.dLossReal += self.vanillaDiscriminatorLoss(predReal, predFake);
			const float eps = 1e-7f;
			const float sm = self.config.labelSmoothing; // one-sided: real target = 1-sm
			float pR = (predReal < eps) ? eps : ((predReal > 1.0f - eps) ? 1.0f - eps : predReal);
			float pF = (predFake < eps) ? eps : ((predFake > 1.0f - eps) ? 1.0f - eps : predFake);
			float dLdReal = -(1.0f - sm) / pR + sm / (1.0f - pR); // smoothed real target
			float dLdFake = 1.0f / (1.0f - pF); // dL/dp for -log(1-p)
			if (self.config.archType == GANConfig::GAN_DFF)
			{
				self.dffBackward(self.discriminator, ctx.discActReal, &dLdReal, 1u, NULL,
				                 ctx.discGrads, ctx.scratchDeltaLocal, true);
				self.dffBackward(self.discriminator, ctx.discActFake, &dLdFake, 1u, NULL,
				                 ctx.discGrads, ctx.scratchDeltaLocal, true);
			}
			else
			{
				self.cnnBackward(self.discriminator, realRow, &dLdReal, 1u, NULL, ctx.discGrads);
				self.cnnBackward(self.discriminator, &fake[0], &dLdFake, 1u, NULL, ctx.discGrads);
			}
		}
		else if (self.config.lossType == GANConfig::GAN_LSGAN)
		{
			ctx.dLossReal += self.lsganDiscriminatorLoss(predReal, predFake);
			const float sm = self.config.labelSmoothing;
			float dLdReal = predReal - (1.0f - sm); // d/dD of 0.5*(D-target)^2
			float dLdFake = predFake;                // d/dD of 0.5*D^2
			if (self.config.archType == GANConfig::GAN_DFF)
			{
				self.dffBackward(self.discriminator, ctx.discActReal, &dLdReal, 1u, NULL,
				                 ctx.discGrads, ctx.scratchDeltaLocal, false);
				self.dffBackward(self.discriminator, ctx.discActFake, &dLdFake, 1u, NULL,
				                 ctx.discGrads, ctx.scratchDeltaLocal, false);
			}
			else
			{
				self.cnnBackward(self.discriminator, realRow, &dLdReal, 1u, NULL, ctx.discGrads, NULL, false);
				self.cnnBackward(self.discriminator, &fake[0], &dLdFake, 1u, NULL, ctx.discGrads, NULL, false);
			}
		}
		else // WGAN-GP
		{
			ctx.dLossReal += self.wganDiscriminatorLoss(predReal, predFake);
			ctx.wasserstein += predReal - predFake;
			float dLdReal = -1.0f;
			float dLdFake = 1.0f;
			if (self.config.archType == GANConfig::GAN_DFF)
			{
				self.dffBackward(self.discriminator, ctx.discActReal, &dLdReal, 1u, NULL,
				                 ctx.discGrads, ctx.scratchDeltaLocal, false);
				self.dffBackward(self.discriminator, ctx.discActFake, &dLdFake, 1u, NULL,
				                 ctx.discGrads, ctx.scratchDeltaLocal, false);
			}
			else
			{
				self.cnnBackward(self.discriminator, realRow, &dLdReal, 1u, NULL, ctx.discGrads, NULL, false);
				self.cnnBackward(self.discriminator, &fake[0], &dLdFake, 1u, NULL, ctx.discGrads, NULL, false);
			}
			// Gradient penalty (monitoring only; Lipschitz enforced via spectral norm)
			ctx.discardDiscGrads.zero();
			float gp = self.computeGradientPenalty(self.discriminator, realRow, &fake[0],
			                                       d.dataDim, d.gpEps[s],
			                                       ctx.discardDiscGrads, ctx.scratchDeltaLocal);
			ctx.dLossReal += self.config.gpLambda * gp;
		}

		// InfoGAN: Q-head on disc penultimate layer
		if (d.hasInfo)
		{
			std::vector<float>& penultAct = ctx.penultActBuf;
			if (self.config.archType == GANConfig::GAN_DFF)
			{
				const unsigned int numDiscLayers = static_cast<unsigned int>(self.discriminator.tensorDff.sizes.size());
				std::memcpy(&penultAct[0], &ctx.discActFake[numDiscLayers - 2u][0], d.penultDim * sizeof(float));
			}
			else
			{
				// CNN: use cached FC activations from disc forward
				const unsigned int numCnnFC = static_cast<unsigned int>(ctx.cnnFcActFake.size());
				const std::vector<float>& src = (numCnnFC >= 2u) ? ctx.cnnFcActFake[numCnnFC - 2u] : ctx.cnnFcActFake[0];
				std::memcpy(&penultAct[0], &src[0], d.penultDim * sizeof(float));
			}

			self.qHeadForward(self.qHead, &penultAct[0], d.penultDim, ctx.qOut);
			float miLoss = self.computeInfoLoss(ctx.qOut, d.catCodes[s], d.contCodes[s], ctx.qGrad);
			ctx.infoLoss += miLoss;

			if (d.numCat > 0u)
			{
				unsigned int predCat = 0u;
				float bestLogit = ctx.qOut[0];
				for (unsigned int c = 1; c < d.numCat; ++c)
					if (ctx.qOut[c] > bestLogit) { bestLogit = ctx.qOut[c]; predCat = c; }
				unsigned int trueCat = 0u;
				for (unsigned int c = 0; c < d.numCat; ++c)
					if (d.catCodes[s][c] > 0.5f) { trueCat = c; break; }
				if (predCat == trueCat) ++ctx.catCorrect;
				++ctx.catTotal;
			}

			for (unsigned int i = 0; i < ctx.qGrad.size(); ++i) ctx.qGrad[i] *= d.infoLambda;
			self.qHeadBackward(self.qHead, &penultAct[0], &ctx.qGrad[0], ctx.sharedGrad, ctx.qGrads);

			// Backprop Q gradient through discriminator
			if (self.config.archType == GANConfig::GAN_DFF)
			{
				const unsigned int numDiscLayers = static_cast<unsigned int>(self.discriminator.tensorDff.sizes.size());
				const unsigned int lastT = static_cast<unsigned int>(self.discriminator.tensorDff.T.size());
				if (lastT >= 1u)
				{
					const NNetwork::TensorDFFState& dff = self.discriminator.tensorDff;
					const NNInfo* skel = self.discriminator.skeleton;
					ctx.qDelta.resize(numDiscLayers);
					for (unsigned int li = 0; li < numDiscLayers; ++li)
						std::memset(&ctx.qDelta[li][0], 0, ctx.qDelta[li].size() * sizeof(float));
					for (unsigned int i = 0; i < d.penultDim && i < ctx.sharedGrad.size(); ++i)
					{
						if (numDiscLayers >= 3u)
						{
							const unsigned int tIdx = numDiscLayers - 3u;
							const int actFx = skel->getActivationType(tIdx);
							const float actParam = skel->getActivationParam(tIdx);
							ctx.qDelta[numDiscLayers - 2u][i] = ctx.sharedGrad[i] * GMath::activationErrDer(ctx.discActFake[numDiscLayers - 2u][i], actFx, actParam);
						}
						else
							ctx.qDelta[numDiscLayers - 2u][i] = ctx.sharedGrad[i];
					}
					for (int li = static_cast<int>(numDiscLayers) - 3; li >= 1; --li)
					{
						const unsigned int l = static_cast<unsigned int>(li);
						const NNetwork::TensorDFFState::Transition& nextTr = dff.T[l];
						for (unsigned int i = 0; i < dff.sizes[l]; ++i)
						{
							float sum = 0.0f;
							for (unsigned int j = 0; j < nextTr.out; ++j)
								sum += ctx.qDelta[l + 1u][j] * nextTr.W[static_cast<size_t>(j) * nextTr.in + i];
							const int actFx = skel->getActivationType(l - 1u);
							const float actParam = skel->getActivationParam(l - 1u);
							ctx.qDelta[l][i] = sum * GMath::activationErrDer(ctx.discActFake[l][i], actFx, actParam);
						}
					}
					// Accumulate Q gradient into ctx.discGrads instead of network
					for (unsigned int t = 0; t < lastT - 1u; ++t)
					{
						const NNetwork::TensorDFFState::Transition& tr = dff.T[t];
						for (unsigned int j = 0; j < tr.out; ++j)
						{
							const float delta = ctx.qDelta[t + 1u][j];
							ctx.discGrads.dffGBias[t][j] += delta;
							const size_t rowOff = static_cast<size_t>(j) * tr.in;
							for (unsigned int i = 0; i < tr.in; ++i)
								ctx.discGrads.dffGW[t][rowOff + i] += delta * ctx.discActFake[t][i];
						}
					}
				}
			}
			else
			{
				// CNN: backprop Q-head gradient through disc
				float zeroGrad = 0.0f;
				self.cnnBackward(self.discriminator, &fake[0], &zeroGrad, 1u, NULL,
				                 ctx.discGrads, &ctx.sharedGrad);
			}
		}
	}
}

// ============================================================
// Parallel Generator Training Body
// ============================================================

void GAN::genTrainBody(void* userData, unsigned int begin, unsigned int end)
{
	GenTrainData& d = *static_cast<GenTrainData*>(userData);
	const unsigned int tid = d.beginToTid[begin];
	GANThreadCtx& ctx = d.ctxs[tid];
	GAN& self = *d.self;

	for (unsigned int s = begin; s < end; ++s)
	{
		// Build genInput from pre-generated noise + codes
		std::memcpy(&ctx.genInput[0], &d.noise[s][0], self.config.noiseDim * sizeof(float));
		if (d.numCat > 0u)
		{
			for (unsigned int c = 0; c < d.numCat; ++c)
				ctx.genInput[self.config.noiseDim + c] = d.catCodes[s][c] * d.catScale;
		}
		if (d.numCont > 0u)
			std::memcpy(&ctx.genInput[self.config.noiseDim + d.numCat], &d.contCodes[s][0], d.numCont * sizeof(float));

		// Generator forward
		if (d.genIsDeconv)
		{
			self.deconvForward(self.generator, &ctx.genInput[0], d.genInputDim, ctx.deconvOut, &ctx.deconvScratch, tid * 1000003u + s * 7u + 1u,
			                   &ctx.deconvArena);
		}
		else if (d.hasStyle)
		{
			self.dffForward(self.mappingNet, &d.noise[s][0], self.config.noiseDim, ctx.mapAct);
			ctx.wVec = ctx.mapAct.back();
			self.dffForwardStyled(self.generator, &ctx.genInput[0], d.genInputDim, ctx.wVec,
			                      self.styleAffines, self.noiseScales,
			                      ctx.genAct, ctx.sXNorms, ctx.sMeans, ctx.sInvStds, ctx.sNoiseVecs);
		}
		else
		{
			LayerNormParams* lnp = d.hasLN ? &d.lnScratch[tid] : NULL;
			self.dffForward(self.generator, &ctx.genInput[0], d.genInputDim, ctx.genAct,
			                self.config.generatorOutputSigmoid, lnp);
		}
		const std::vector<float>& fake = d.genIsDeconv ? ctx.deconvOut : ctx.genAct.back();

		// Discriminator forward on fake
		const bool discSig = (self.config.lossType == GANConfig::GAN_VANILLA);
		float predFake = 0.0f;
		if (self.config.archType == GANConfig::GAN_DFF)
		{
			self.dffForward(self.discriminator, &fake[0], d.dataDim, ctx.discActFake, discSig);
			predFake = ctx.discActFake.back()[0];
		}
		else
		{
			if (d.hasInfo)
				self.cnnForward(self.discriminator, &fake[0], ctx.cnnOutFake, discSig, &ctx.cnnFcActFake);
			else
				self.cnnForward(self.discriminator, &fake[0], ctx.cnnOutFake, discSig);
			predFake = ctx.cnnOutFake[0];
		}

		// Generator loss
		float dLdFakeForGen;
		if (self.config.lossType == GANConfig::GAN_VANILLA)
		{
			// predFake is already sigmoid output from forward
			ctx.gLoss += self.vanillaGeneratorLoss(predFake);
			// Combined gradient dL/dz = -(1-p), avoids numerical issues with
			// -1/p_clamped * sigmoidDeriv(p_actual) when p_actual << eps
			dLdFakeForGen = -(1.0f - predFake);
		}
		else if (self.config.lossType == GANConfig::GAN_LSGAN)
		{
			ctx.gLoss += self.lsganGeneratorLoss(predFake);
			dLdFakeForGen = predFake - 1.0f; // d/dD of 0.5*(D-1)^2
		}
		else
		{
			ctx.gLoss += self.wganGeneratorLoss(predFake);
			dLdFakeForGen = -1.0f;
		}

		// Disc backward -> dFake (disc grads discarded)
		ctx.discardDiscGrads.zero();
		if (self.config.archType == GANConfig::GAN_DFF)
		{
			self.dffBackward(self.discriminator, ctx.discActFake, &dLdFakeForGen, 1u, &ctx.dFake,
			                 ctx.discardDiscGrads, ctx.scratchDeltaLocal);
		}
		else
		{
			self.cnnBackward(self.discriminator, &fake[0], &dLdFakeForGen, 1u, &ctx.dFake,
			                 ctx.discardDiscGrads, NULL, false);
		}

		// InfoGAN: Q-head on discriminator features (proper InfoGAN architecture)
		// Routes info gradient through disc -> Q -> back through disc -> dFake
		if (d.hasInfo)
		{
			// Get penultimate discriminator activations (already computed from disc forward above)
			std::vector<float>& penultAct = ctx.penultActBuf;
			if (self.config.archType == GANConfig::GAN_DFF)
			{
				const unsigned int numDiscLayers = static_cast<unsigned int>(self.discriminator.tensorDff.sizes.size());
				std::memcpy(&penultAct[0], &ctx.discActFake[numDiscLayers - 2u][0], d.penultDim * sizeof(float));
			}
			else
			{
				// CNN: use cached FC activations from disc forward
				const unsigned int numCnnFC = static_cast<unsigned int>(ctx.cnnFcActFake.size());
				const std::vector<float>& src = (numCnnFC >= 2u) ? ctx.cnnFcActFake[numCnnFC - 2u] : ctx.cnnFcActFake[0];
				std::memcpy(&penultAct[0], &src[0], d.penultDim * sizeof(float));
			}

			// Forward through disc-side Q-head
			self.qHeadForward(self.qHead, &penultAct[0], d.penultDim, ctx.qOut);
			float miLoss = self.computeInfoLoss(ctx.qOut, d.catCodes[s], d.contCodes[s], ctx.qGrad);
			ctx.infoLoss += miLoss;
			for (unsigned int i = 0; i < ctx.qGrad.size(); ++i) ctx.qGrad[i] *= d.infoLambda;

			// Q-head backward: compute gradient on penultimate activations only
			// (no Q-head weight accumulation — Q is updated during disc training)
			ctx.sharedGrad.assign(self.qHead.sharedDim, 0.0f);
			for (unsigned int j = 0; j < self.qHead.qOutDim; ++j)
			{
				const float dj = ctx.qGrad[j];
				const size_t rowOff = static_cast<size_t>(j) * self.qHead.sharedDim;
				for (unsigned int i = 0; i < self.qHead.sharedDim; ++i)
					ctx.sharedGrad[i] += dj * self.qHead.W[rowOff + i];
			}

			// Backprop Q gradient through discriminator to get dFake (discard disc grads)
			if (self.config.archType == GANConfig::GAN_DFF)
			{
				const unsigned int numDiscLayers = static_cast<unsigned int>(self.discriminator.tensorDff.sizes.size());
				const unsigned int lastT = static_cast<unsigned int>(self.discriminator.tensorDff.T.size());
				if (lastT >= 1u)
				{
					const NNetwork::TensorDFFState& dff = self.discriminator.tensorDff;
					const NNInfo* skel = self.discriminator.skeleton;
					ctx.qDelta.resize(numDiscLayers);
					for (unsigned int li = 0; li < numDiscLayers; ++li)
						std::memset(&ctx.qDelta[li][0], 0, ctx.qDelta[li].size() * sizeof(float));

					// Penultimate layer: apply activation derivative
					for (unsigned int i = 0; i < d.penultDim && i < ctx.sharedGrad.size(); ++i)
					{
						if (numDiscLayers >= 3u)
						{
							const unsigned int tIdx = numDiscLayers - 3u;
							const int actFx = skel->getActivationType(tIdx);
							const float actParam = skel->getActivationParam(tIdx);
							ctx.qDelta[numDiscLayers - 2u][i] = ctx.sharedGrad[i] * GMath::activationErrDer(ctx.discActFake[numDiscLayers - 2u][i], actFx, actParam);
						}
						else
							ctx.qDelta[numDiscLayers - 2u][i] = ctx.sharedGrad[i];
					}

					// Backward through hidden layers (no parameter gradient accumulation)
					for (int li = static_cast<int>(numDiscLayers) - 3; li >= 1; --li)
					{
						const unsigned int l = static_cast<unsigned int>(li);
						const NNetwork::TensorDFFState::Transition& nextTr = dff.T[l];
						for (unsigned int i = 0; i < dff.sizes[l]; ++i)
						{
							float sum = 0.0f;
							for (unsigned int j = 0; j < nextTr.out; ++j)
								sum += ctx.qDelta[l + 1u][j] * nextTr.W[static_cast<size_t>(j) * nextTr.in + i];
							const int actFx = skel->getActivationType(l - 1u);
							const float actParam = skel->getActivationParam(l - 1u);
							ctx.qDelta[l][i] = sum * GMath::activationErrDer(ctx.discActFake[l][i], actFx, actParam);
						}
					}

					// Input gradient: add Q-path contribution to dFake
					const NNetwork::TensorDFFState::Transition& tr0 = dff.T[0];
					for (unsigned int i = 0; i < tr0.in && i < d.dataDim; ++i)
					{
						float sum = 0.0f;
						for (unsigned int j = 0; j < tr0.out; ++j)
							sum += ctx.qDelta[1][j] * tr0.W[static_cast<size_t>(j) * tr0.in + i];
						ctx.dFake[i] += sum;
					}
				}
			}
			else
			{
				// CNN: backprop Q gradient through disc, capture input gradient
				std::vector<float>& qInfoDFake = ctx.qInfoDFakeBuf;
				float zeroGrad = 0.0f;
				self.cnnBackward(self.discriminator, &fake[0], &zeroGrad, 1u, &qInfoDFake,
				                 ctx.discardDiscGrads, &ctx.sharedGrad, false);
				for (unsigned int i = 0; i < d.dataDim && i < qInfoDFake.size(); ++i)
					ctx.dFake[i] += qInfoDFake[i];
			}
		}

		// Mode-seeking diversity loss: penalise identical outputs for different noise vectors.
		// Compare current fake to previous sample from this thread (no extra forward pass).
		if (d.divLambda > 0.0f && s > begin && !ctx.prevFake.empty())
		{
			float dOut = 0.0f, dIn = 0.0f;
			for (unsigned int i = 0; i < d.dataDim; ++i)
				dOut += fabsf(fake[i] - ctx.prevFake[i]);
			for (unsigned int i = 0; i < self.config.noiseDim; ++i)
				dIn += fabsf(d.noise[s][i] - d.noise[s - 1u][i]);
			const float invDIn = 1.0f / (dIn + 1e-6f);
			ctx.divLoss += -dOut * invDIn;

			// Gradient w.r.t. fake: push away from prevFake
			const float scale = d.divLambda * invDIn;
			for (unsigned int i = 0; i < d.dataDim; ++i)
			{
				const float diff = fake[i] - ctx.prevFake[i];
				const float sgn = (diff > 0.0f) ? 1.0f : ((diff < 0.0f) ? -1.0f : 0.0f);
				ctx.dFake[i] -= sgn * scale;
			}
		}
		if (d.divLambda > 0.0f)
		{
			ctx.prevFake.resize(d.dataDim);
			std::memcpy(&ctx.prevFake[0], &fake[0], d.dataDim * sizeof(float));
		}

		// Generator backward
		if (d.genIsDeconv)
		{
			self.deconvBackward(self.generator, ctx.deconvScratch, &ctx.dFake[0], d.dataDim, NULL,
			                    ctx.genGrads, &ctx.deconvArena);
		}
		else if (d.hasStyle)
		{
			self.dffBackwardStyled(self.generator, ctx.genAct, &ctx.dFake[0], d.dataDim,
			                       ctx.wVec, ctx.sXNorms, ctx.sMeans, ctx.sInvStds, ctx.sNoiseVecs,
			                       self.styleAffines, ctx.gNoiseScalesLocal, ctx.dWVec,
			                       ctx.genGrads, ctx.styleGW, ctx.styleGBias, ctx.scratchDeltaLocal);
			// Mapping backward
			self.dffBackward(self.mappingNet, ctx.mapAct, &ctx.dWVec[0],
			                 static_cast<unsigned int>(ctx.dWVec.size()), NULL,
			                 ctx.mappingGrads, ctx.scratchDeltaLocal);
		}
		else if (d.hasLN)
		{
			self.dffBackward(self.generator, ctx.genAct, &ctx.dFake[0], d.dataDim, NULL,
			                 ctx.genGrads, ctx.scratchDeltaLocal,
			                 false, // straight-through: skip sigmoid deriv for generator output
			                 &d.lnScratch[tid], ctx.lnGGamma, ctx.lnGBeta);
		}
		else
		{
			self.dffBackward(self.generator, ctx.genAct, &ctx.dFake[0], d.dataDim, NULL,
			                 ctx.genGrads, ctx.scratchDeltaLocal,
			                 false); // straight-through
		}
	}
}

// ============================================================
// CycleGAN Parallel Discriminator Body
// ============================================================

void GAN::cycleDiscBody(void* userData, unsigned int begin, unsigned int end)
{
	CycleDiscData& d = *static_cast<CycleDiscData*>(userData);
	const unsigned int tid = d.beginToTid[begin];
	GANThreadCtx& ctx = d.ctxs[tid];
	GAN& self = *d.self;

	// Select which network set to use based on isDiscA
	NNetwork& disc = d.isDiscA ? self.discriminator : self.discriminatorB;
	NNetwork& gen = d.isDiscA ? self.generatorBA : self.generator;
	NNetwork& mapNet = d.isDiscA ? self.mappingNetBA : self.mappingNet;
	const std::vector<StyleAffine>& affines = d.isDiscA ? self.styleAffinesBA : self.styleAffines;
	const std::vector<float>& scales = d.isDiscA ? self.noiseScalesBA : self.noiseScales;
	QNetworkHead& qh = d.isDiscA ? self.qHead : self.qHeadB;
	GradientBuffer& discGradBuf = d.isDiscA ? ctx.discGrads : ctx.discBGrads;
	GradientBuffer& qGradBuf = d.isDiscA ? ctx.qGrads : ctx.qBGrads;
	GradientBuffer& discardBuf = d.isDiscA ? ctx.discardDiscGrads : ctx.discardDiscBGrads;
	LayerNormParams* genLnp = d.isDiscA ? self.genBALN() : self.genLN();

	// Use genInput/genAct for D_A (genBA), genBAInput/genBAAct for D_B (genAB)
	std::vector<float>& genInputBuf = d.isDiscA ? ctx.genBAInput : ctx.genInput;
	std::vector<std::vector<float> >& genActBuf = d.isDiscA ? ctx.genBAAct : ctx.genAct;
	std::vector<float>& deconvOutBuf = d.isDiscA ? ctx.deconvOutBA : ctx.deconvOut;
	std::vector<std::vector<float> >& deconvScrBuf = d.isDiscA ? ctx.deconvScratchBA : ctx.deconvScratch;
	std::vector<std::vector<float> >& mapActBuf = d.isDiscA ? ctx.mapBAAct : ctx.mapAct;

	// Use discActReal/discActFake for D_A, discBActReal/discBActFake for D_B
	std::vector<std::vector<float> >& discActRealBuf = d.isDiscA ? ctx.discActReal : ctx.discBActReal;
	std::vector<std::vector<float> >& discActFakeBuf = d.isDiscA ? ctx.discActFake : ctx.discBActFake;
	std::vector<float>& cnnOutRealBuf = d.isDiscA ? ctx.cnnOutReal : ctx.cnnOutBReal;
	std::vector<float>& cnnOutFakeBuf = d.isDiscA ? ctx.cnnOutFake : ctx.cnnOutBFake;

	for (unsigned int s = begin; s < end; ++s)
	{
		// Source domain (opposite of disc domain) provides input to generator
		const float* srcRow = d.isDiscA ? d.realPtrsB[s] : d.realPtrsA[s];
		// Target domain provides real data for disc
		const float* tgtRow = d.isDiscA ? d.realPtrsA[s] : d.realPtrsB[s];

		// Build generator input: [srcRow | codes?]
		std::memcpy(&genInputBuf[0], srcRow, d.srcDim * sizeof(float));
		if (d.numCat > 0u)
			std::memcpy(&genInputBuf[d.srcDim], &d.catCodes[s][0], d.numCat * sizeof(float));
		if (d.numCont > 0u)
			std::memcpy(&genInputBuf[d.srcDim + d.numCat], &d.contCodes[s][0], d.numCont * sizeof(float));

		// Generator forward to produce fake in target domain
		if (d.genIsDeconv)
		{
			self.deconvForward(gen, &genInputBuf[0], d.genInputDim, deconvOutBuf, &deconvScrBuf, tid * 1000003u + s * 7u + 1u);
		}
		else if (d.hasStyle)
		{
			self.dffForward(mapNet, &d.styleNoise[s][0], self.config.noiseDim, mapActBuf);
			const std::vector<float>& w = mapActBuf.back();
			self.dffForwardStyled(gen, &genInputBuf[0], d.genInputDim, w,
			                      affines, scales,
			                      genActBuf, ctx.sXNorms, ctx.sMeans, ctx.sInvStds, ctx.sNoiseVecs);
		}
		else
		{
			self.dffForward(gen, &genInputBuf[0], d.genInputDim, genActBuf,
			                self.config.generatorOutputSigmoid, genLnp);
		}
		const std::vector<float>& fake = d.genIsDeconv ? deconvOutBuf : genActBuf.back();

		// Discriminator on real
		const bool discSig = (self.config.lossType == GANConfig::GAN_VANILLA);
		float predReal = 0.0f;
		if (self.config.archType == GANConfig::GAN_DFF)
		{
			self.dffForward(disc, tgtRow, d.tgtDim, discActRealBuf, discSig);
			predReal = discActRealBuf.back()[0];
		}
		else
		{
			self.cnnForward(disc, tgtRow, cnnOutRealBuf, discSig);
			predReal = cnnOutRealBuf[0];
		}

		// Discriminator on fake
		float predFake = 0.0f;
		if (self.config.archType == GANConfig::GAN_DFF)
		{
			self.dffForward(disc, &fake[0], d.tgtDim, discActFakeBuf, discSig);
			predFake = discActFakeBuf.back()[0];
		}
		else
		{
			self.cnnForward(disc, &fake[0], cnnOutFakeBuf, discSig);
			predFake = cnnOutFakeBuf[0];
		}

		// Adversarial loss + backward
		if (self.config.lossType == GANConfig::GAN_VANILLA)
		{
			// predReal/predFake are already sigmoid outputs from forward
			ctx.dLossReal += self.vanillaDiscriminatorLoss(predReal, predFake);
			const float eps = 1e-7f;
			const float sm = self.config.labelSmoothing;
			float pR = (predReal < eps) ? eps : ((predReal > 1.0f - eps) ? 1.0f - eps : predReal);
			float pF = (predFake < eps) ? eps : ((predFake > 1.0f - eps) ? 1.0f - eps : predFake);
			float dLdReal = -(1.0f - sm) / pR + sm / (1.0f - pR); // smoothed real target
			float dLdFake = 1.0f / (1.0f - pF); // dL/dp for -log(1-p)
			if (self.config.archType == GANConfig::GAN_DFF)
			{
				self.dffBackward(disc, discActRealBuf, &dLdReal, 1u, NULL,
				                 discGradBuf, ctx.scratchDeltaLocal, true);
				self.dffBackward(disc, discActFakeBuf, &dLdFake, 1u, NULL,
				                 discGradBuf, ctx.scratchDeltaLocal, true);
			}
			else
			{
				self.cnnBackward(disc, tgtRow, &dLdReal, 1u, NULL, discGradBuf);
				self.cnnBackward(disc, &fake[0], &dLdFake, 1u, NULL, discGradBuf);
			}
		}
		else if (self.config.lossType == GANConfig::GAN_LSGAN)
		{
			ctx.dLossReal += self.lsganDiscriminatorLoss(predReal, predFake);
			const float sm = self.config.labelSmoothing;
			float dLdReal = predReal - (1.0f - sm);
			float dLdFake = predFake;
			if (self.config.archType == GANConfig::GAN_DFF)
			{
				self.dffBackward(disc, discActRealBuf, &dLdReal, 1u, NULL,
				                 discGradBuf, ctx.scratchDeltaLocal, false);
				self.dffBackward(disc, discActFakeBuf, &dLdFake, 1u, NULL,
				                 discGradBuf, ctx.scratchDeltaLocal, false);
			}
			else
			{
				self.cnnBackward(disc, tgtRow, &dLdReal, 1u, NULL, discGradBuf, NULL, false);
				self.cnnBackward(disc, &fake[0], &dLdFake, 1u, NULL, discGradBuf, NULL, false);
			}
		}
		else // WGAN-GP
		{
			ctx.dLossReal += self.wganDiscriminatorLoss(predReal, predFake);
			ctx.wasserstein += predReal - predFake;
			float dLdReal = -1.0f;
			float dLdFake = 1.0f;
			if (self.config.archType == GANConfig::GAN_DFF)
			{
				self.dffBackward(disc, discActRealBuf, &dLdReal, 1u, NULL,
				                 discGradBuf, ctx.scratchDeltaLocal, false);
				self.dffBackward(disc, discActFakeBuf, &dLdFake, 1u, NULL,
				                 discGradBuf, ctx.scratchDeltaLocal, false);
			}
			else
			{
				self.cnnBackward(disc, tgtRow, &dLdReal, 1u, NULL, discGradBuf, NULL, false);
				self.cnnBackward(disc, &fake[0], &dLdFake, 1u, NULL, discGradBuf, NULL, false);
			}
			// Gradient penalty (monitoring only; Lipschitz enforced via spectral norm)
			discardBuf.zero();
			float gp = self.computeGradientPenalty(disc, tgtRow, &fake[0],
			                                       d.tgtDim, d.gpEps[s],
			                                       discardBuf, ctx.scratchDeltaLocal);
			ctx.dLossReal += self.config.gpLambda * gp;
		}

		// InfoGAN: Q-head on disc penultimate layer
		if (d.hasInfo)
		{
			std::vector<float> penultAct;
			if (self.config.archType == GANConfig::GAN_DFF)
			{
				const unsigned int numDiscLayers = static_cast<unsigned int>(disc.tensorDff.sizes.size());
				penultAct = discActFakeBuf[numDiscLayers - 2u];
			}
			else
			{
				std::vector<float> discOutTmp;
				std::vector<std::vector<float> > cnnFcAct;
				self.cnnForward(disc, &fake[0], discOutTmp, discSig, &cnnFcAct);
				const unsigned int numCnnFC = static_cast<unsigned int>(cnnFcAct.size());
				penultAct = (numCnnFC >= 2u) ? cnnFcAct[numCnnFC - 2u] : cnnFcAct[0];
			}

			self.qHeadForward(qh, &penultAct[0], d.penultDim, ctx.qOut);
			float miLoss = self.computeInfoLoss(ctx.qOut, d.catCodes[s], d.contCodes[s], ctx.qGrad);
			ctx.infoLoss += miLoss;

			if (d.numCat > 0u)
			{
				unsigned int predCat = 0u;
				float bestLogit = ctx.qOut[0];
				for (unsigned int c = 1; c < d.numCat; ++c)
					if (ctx.qOut[c] > bestLogit) { bestLogit = ctx.qOut[c]; predCat = c; }
				unsigned int trueCat = 0u;
				for (unsigned int c = 0; c < d.numCat; ++c)
					if (d.catCodes[s][c] > 0.5f) { trueCat = c; break; }
				if (predCat == trueCat) ++ctx.catCorrect;
				++ctx.catTotal;
			}

			for (unsigned int i = 0; i < ctx.qGrad.size(); ++i) ctx.qGrad[i] *= d.infoLambda;
			self.qHeadBackward(qh, &penultAct[0], &ctx.qGrad[0], ctx.sharedGrad, qGradBuf);

			// Backprop Q gradient through discriminator
			if (self.config.archType == GANConfig::GAN_DFF)
			{
				const unsigned int numDiscLayers = static_cast<unsigned int>(disc.tensorDff.sizes.size());
				const unsigned int lastT = static_cast<unsigned int>(disc.tensorDff.T.size());
				if (lastT >= 1u)
				{
					const NNetwork::TensorDFFState& dff = disc.tensorDff;
					const NNInfo* skel = disc.skeleton;
					ctx.qDelta.resize(numDiscLayers);
					for (unsigned int li = 0; li < numDiscLayers; ++li)
						ctx.qDelta[li].assign(dff.sizes[li], 0.0f);
					for (unsigned int i = 0; i < d.penultDim && i < ctx.sharedGrad.size(); ++i)
					{
						if (numDiscLayers >= 3u)
						{
							const unsigned int tIdx = numDiscLayers - 3u;
							const int actFx = skel->getActivationType(tIdx);
							const float actParam = skel->getActivationParam(tIdx);
							ctx.qDelta[numDiscLayers - 2u][i] = ctx.sharedGrad[i] * GMath::activationErrDer(discActFakeBuf[numDiscLayers - 2u][i], actFx, actParam);
						}
						else
							ctx.qDelta[numDiscLayers - 2u][i] = ctx.sharedGrad[i];
					}
					for (int li = static_cast<int>(numDiscLayers) - 3; li >= 1; --li)
					{
						const unsigned int l = static_cast<unsigned int>(li);
						const NNetwork::TensorDFFState::Transition& nextTr = dff.T[l];
						for (unsigned int i = 0; i < dff.sizes[l]; ++i)
						{
							float sum = 0.0f;
							for (unsigned int j = 0; j < nextTr.out; ++j)
								sum += ctx.qDelta[l + 1u][j] * nextTr.W[static_cast<size_t>(j) * nextTr.in + i];
							const int actFx = skel->getActivationType(l - 1u);
							const float actParam = skel->getActivationParam(l - 1u);
							ctx.qDelta[l][i] = sum * GMath::activationErrDer(discActFakeBuf[l][i], actFx, actParam);
						}
					}
					// Accumulate Q gradient into discGradBuf
					for (unsigned int t = 0; t < lastT - 1u; ++t)
					{
						const NNetwork::TensorDFFState::Transition& tr = dff.T[t];
						for (unsigned int j = 0; j < tr.out; ++j)
						{
							const float delta = ctx.qDelta[t + 1u][j];
							discGradBuf.dffGBias[t][j] += delta;
							const size_t rowOff = static_cast<size_t>(j) * tr.in;
							for (unsigned int i = 0; i < tr.in; ++i)
								discGradBuf.dffGW[t][rowOff + i] += delta * discActFakeBuf[t][i];
						}
					}
				}
			}
			else
			{
				float zeroGrad = 0.0f;
				self.cnnBackward(disc, &fake[0], &zeroGrad, 1u, NULL,
				                 discGradBuf, &ctx.sharedGrad);
			}
		}
	}
}

// ============================================================
// CycleGAN Parallel Generator Body
// ============================================================

void GAN::cycleGenBody(void* userData, unsigned int begin, unsigned int end)
{
	CycleGenData& d = *static_cast<CycleGenData*>(userData);
	const unsigned int tid = d.beginToTid[begin];
	GANThreadCtx& ctx = d.ctxs[tid];
	GAN& self = *d.self;

	for (unsigned int s = begin; s < end; ++s)
	{
		const float* realA = d.realPtrsA[s];
		const float* realB = d.realPtrsB[s];

		// Build genAB input [realA | codes?]
		std::memcpy(&ctx.genInput[0], realA, d.dimA * sizeof(float));
		if (d.numCat > 0u) std::memcpy(&ctx.genInput[d.dimA], &d.catCodesAB[s][0], d.numCat * sizeof(float));
		if (d.numCont > 0u) std::memcpy(&ctx.genInput[d.dimA + d.numCat], &d.contCodesAB[s][0], d.numCont * sizeof(float));

		// Build genBA input [realB | codes?]
		std::memcpy(&ctx.genBAInput[0], realB, d.dimB * sizeof(float));
		if (d.numCat > 0u) std::memcpy(&ctx.genBAInput[d.dimB], &d.catCodesBA[s][0], d.numCat * sizeof(float));
		if (d.numCont > 0u) std::memcpy(&ctx.genBAInput[d.dimB + d.numCat], &d.contCodesBA[s][0], d.numCont * sizeof(float));

		// Forward passes for fakeB and fakeA
		if (d.genIsDeconv)
		{
			self.deconvForward(self.generator, &ctx.genInput[0], d.genABInputDim, ctx.deconvOut, &ctx.deconvScratch, tid * 1000003u + s * 7u + 1u);
			self.deconvForward(self.generatorBA, &ctx.genBAInput[0], d.genBAInputDim, ctx.deconvOutBA, &ctx.deconvScratchBA, tid * 1000003u + s * 7u + 2u);
		}
		else if (d.hasStyle)
		{
			self.dffForward(self.mappingNet, &d.styleNoiseAB[s][0], self.config.noiseDim, ctx.mapAct);
			ctx.wVec = ctx.mapAct.back();
			self.dffForwardStyled(self.generator, &ctx.genInput[0], d.genABInputDim, ctx.wVec,
			                      self.styleAffines, self.noiseScales,
			                      ctx.genAct, ctx.sXNorms, ctx.sMeans, ctx.sInvStds, ctx.sNoiseVecs);

			self.dffForward(self.mappingNetBA, &d.styleNoiseBA[s][0], self.config.noiseDim, ctx.mapBAAct);
			ctx.wBAVec = ctx.mapBAAct.back();
			self.dffForwardStyled(self.generatorBA, &ctx.genBAInput[0], d.genBAInputDim, ctx.wBAVec,
			                      self.styleAffinesBA, self.noiseScalesBA,
			                      ctx.genBAAct, ctx.sBAXNorms, ctx.sBAMeans, ctx.sBAInvStds, ctx.sBANoiseVecs);
		}
		else
		{
			LayerNormParams* lnpAB = d.hasLN ? &d.lnScratchAB[tid] : NULL;
			LayerNormParams* lnpBA = d.hasBALN ? &d.lnScratchBA[tid] : NULL;
			self.dffForward(self.generator, &ctx.genInput[0], d.genABInputDim, ctx.genAct,
			                self.config.generatorOutputSigmoid, lnpAB);
			self.dffForward(self.generatorBA, &ctx.genBAInput[0], d.genBAInputDim, ctx.genBAAct,
			                self.config.generatorOutputSigmoid, lnpBA);
		}
		const std::vector<float>& fakeB = d.genIsDeconv ? ctx.deconvOut : ctx.genAct.back();
		const std::vector<float>& fakeA = d.genIsDeconv ? ctx.deconvOutBA : ctx.genBAAct.back();

		// Cycle: rec_A = G_BA(fake_B [| codes]), rec_B = G_AB(fake_A [| codes])
		std::vector<float> recAInput(d.genBAInputDim);
		std::memcpy(&recAInput[0], &fakeB[0], d.dimB * sizeof(float));
		if (d.numCat > 0u) std::memcpy(&recAInput[d.dimB], &d.catCodesBA[s][0], d.numCat * sizeof(float));
		if (d.numCont > 0u) std::memcpy(&recAInput[d.dimB + d.numCat], &d.contCodesBA[s][0], d.numCont * sizeof(float));

		std::vector<float> recBInput(d.genABInputDim);
		std::memcpy(&recBInput[0], &fakeA[0], d.dimA * sizeof(float));
		if (d.numCat > 0u) std::memcpy(&recBInput[d.dimA], &d.catCodesAB[s][0], d.numCat * sizeof(float));
		if (d.numCont > 0u) std::memcpy(&recBInput[d.dimA + d.numCat], &d.contCodesAB[s][0], d.numCont * sizeof(float));

		if (d.genIsDeconv)
		{
			self.deconvForward(self.generatorBA, &recAInput[0], d.genBAInputDim, ctx.deconvOutRecA, &ctx.deconvScratchRecA, tid * 1000003u + s * 7u + 3u);
			self.deconvForward(self.generator, &recBInput[0], d.genABInputDim, ctx.deconvOutRecB, &ctx.deconvScratchRecB, tid * 1000003u + s * 7u + 4u);
		}
		else if (d.hasStyle)
		{
			std::vector<std::vector<float> > mapRecAAct, mapRecBAct;
			self.dffForward(self.mappingNetBA, &d.styleNoiseRecA[s][0], self.config.noiseDim, mapRecAAct);
			std::vector<std::vector<float> > xNTmp;
			std::vector<float> mTmp, iTmp;
			std::vector<std::vector<float> > nTmp;
			self.dffForwardStyled(self.generatorBA, &recAInput[0], d.genBAInputDim, mapRecAAct.back(),
			                      self.styleAffinesBA, self.noiseScalesBA,
			                      ctx.recAAct, xNTmp, mTmp, iTmp, nTmp);

			self.dffForward(self.mappingNet, &d.styleNoiseRecB[s][0], self.config.noiseDim, mapRecBAct);
			self.dffForwardStyled(self.generator, &recBInput[0], d.genABInputDim, mapRecBAct.back(),
			                      self.styleAffines, self.noiseScales,
			                      ctx.recBAct, xNTmp, mTmp, iTmp, nTmp);
		}
		else
		{
			LayerNormParams* lnpBA = d.hasBALN ? &d.lnScratchBA[tid] : NULL;
			LayerNormParams* lnpAB = d.hasLN ? &d.lnScratchAB[tid] : NULL;
			self.dffForward(self.generatorBA, &recAInput[0], d.genBAInputDim, ctx.recAAct,
			                self.config.generatorOutputSigmoid, lnpBA);
			self.dffForward(self.generator, &recBInput[0], d.genABInputDim, ctx.recBAct,
			                self.config.generatorOutputSigmoid, lnpAB);
		}
		const std::vector<float>& recA = d.genIsDeconv ? ctx.deconvOutRecA : ctx.recAAct.back();
		const std::vector<float>& recB = d.genIsDeconv ? ctx.deconvOutRecB : ctx.recBAct.back();

		// Identity: ident_B = G_AB(y_B [| zero_codes]), ident_A = G_BA(x_A [| zero_codes])
		std::vector<float> identBInput(d.genABInputDim);
		std::memcpy(&identBInput[0], realB, d.dimB * sizeof(float));
		if (d.numCat > 0u) std::memset(&identBInput[d.dimB], 0, d.numCat * sizeof(float));
		if (d.numCont > 0u) std::memset(&identBInput[d.dimB + d.numCat], 0, d.numCont * sizeof(float));

		std::vector<float> identAInput(d.genBAInputDim);
		std::memcpy(&identAInput[0], realA, d.dimA * sizeof(float));
		if (d.numCat > 0u) std::memset(&identAInput[d.dimA], 0, d.numCat * sizeof(float));
		if (d.numCont > 0u) std::memset(&identAInput[d.dimA + d.numCat], 0, d.numCont * sizeof(float));

		if (d.genIsDeconv)
		{
			self.deconvForward(self.generator, &identBInput[0], d.genABInputDim, ctx.deconvOutIdentB, &ctx.deconvScratchIdentB, tid * 1000003u + s * 7u + 5u);
			self.deconvForward(self.generatorBA, &identAInput[0], d.genBAInputDim, ctx.deconvOutIdentA, &ctx.deconvScratchIdentA, tid * 1000003u + s * 7u + 6u);
		}
		else if (d.hasStyle)
		{
			std::vector<std::vector<float> > mapIdBAct, mapIdAAct;
			self.dffForward(self.mappingNet, &d.styleNoiseIdentB[s][0], self.config.noiseDim, mapIdBAct);
			std::vector<std::vector<float> > xNTmp;
			std::vector<float> mTmp, iTmp;
			std::vector<std::vector<float> > nTmp;
			self.dffForwardStyled(self.generator, &identBInput[0], d.genABInputDim, mapIdBAct.back(),
			                      self.styleAffines, self.noiseScales,
			                      ctx.identBAct, xNTmp, mTmp, iTmp, nTmp);

			self.dffForward(self.mappingNetBA, &d.styleNoiseIdentA[s][0], self.config.noiseDim, mapIdAAct);
			self.dffForwardStyled(self.generatorBA, &identAInput[0], d.genBAInputDim, mapIdAAct.back(),
			                      self.styleAffinesBA, self.noiseScalesBA,
			                      ctx.identAAct, xNTmp, mTmp, iTmp, nTmp);
		}
		else
		{
			LayerNormParams* lnpAB = d.hasLN ? &d.lnScratchAB[tid] : NULL;
			LayerNormParams* lnpBA = d.hasBALN ? &d.lnScratchBA[tid] : NULL;
			self.dffForward(self.generator, &identBInput[0], d.genABInputDim, ctx.identBAct,
			                self.config.generatorOutputSigmoid, lnpAB);
			self.dffForward(self.generatorBA, &identAInput[0], d.genBAInputDim, ctx.identAAct,
			                self.config.generatorOutputSigmoid, lnpBA);
		}
		const std::vector<float>& identB = d.genIsDeconv ? ctx.deconvOutIdentB : ctx.identBAct.back();
		const std::vector<float>& identA = d.genIsDeconv ? ctx.deconvOutIdentA : ctx.identAAct.back();

		// Adversarial: D_B(fake_B)
		const bool discSig = (self.config.lossType == GANConfig::GAN_VANILLA);
		float predFakeB = 0.0f;
		if (self.config.archType == GANConfig::GAN_DFF)
		{
			self.dffForward(self.discriminatorB, &fakeB[0], d.dimB, ctx.discBFakeAct, discSig);
			predFakeB = ctx.discBFakeAct.back()[0];
		}
		else
		{
			self.cnnForward(self.discriminatorB, &fakeB[0], ctx.cnnOutFakeBGen, discSig);
			predFakeB = ctx.cnnOutFakeBGen[0];
		}

		// Adversarial: D_A(fake_A)
		float predFakeA = 0.0f;
		if (self.config.archType == GANConfig::GAN_DFF)
		{
			self.dffForward(self.discriminator, &fakeA[0], d.dimA, ctx.discAFakeAct, discSig);
			predFakeA = ctx.discAFakeAct.back()[0];
		}
		else
		{
			self.cnnForward(self.discriminator, &fakeA[0], ctx.cnnOutFakeAGen, discSig);
			predFakeA = ctx.cnnOutFakeAGen[0];
		}

		// Generator adversarial loss
		float dLdFakeForGenB = 0.0f, dLdFakeForGenA = 0.0f;
		if (self.config.lossType == GANConfig::GAN_VANILLA)
		{
			// predFakeB/predFakeA are already sigmoid outputs from forward
			ctx.gLossAB += self.vanillaGeneratorLoss(predFakeB);
			ctx.gLossBA += self.vanillaGeneratorLoss(predFakeA);
			// Combined gradient dL/dz = -(1-p), avoids numerical issues with
			// -1/p_clamped * sigmoidDeriv(p_actual) when p_actual << eps
			dLdFakeForGenB = -(1.0f - predFakeB);
			dLdFakeForGenA = -(1.0f - predFakeA);
		}
		else if (self.config.lossType == GANConfig::GAN_LSGAN)
		{
			ctx.gLossAB += self.lsganGeneratorLoss(predFakeB);
			ctx.gLossBA += self.lsganGeneratorLoss(predFakeA);
			dLdFakeForGenB = predFakeB - 1.0f;
			dLdFakeForGenA = predFakeA - 1.0f;
		}
		else // WGAN-GP
		{
			ctx.gLossAB += self.wganGeneratorLoss(predFakeB);
			ctx.gLossBA += self.wganGeneratorLoss(predFakeA);
			dLdFakeForGenB = -1.0f;
			dLdFakeForGenA = -1.0f;
		}

		// Compute losses
		const float cycA = l1Loss(&recA[0], realA, d.dimA);
		const float cycB = l1Loss(&recB[0], realB, d.dimB);
		ctx.cycleLoss += cycA + cycB;

		const float idA = l1Loss(&identA[0], realA, d.dimA);
		const float idB = l1Loss(&identB[0], realB, d.dimB);
		ctx.identityLoss += idA + idB;

		// InfoGAN: Q-head contributions for generator
		std::vector<float> dFakeB_info, dFakeA_info;
		if (d.hasInfo)
		{
			// Q-head B on D_B features from fakeB
			{
				std::vector<float> penultActB;
				if (self.config.archType == GANConfig::GAN_DFF)
				{
					const unsigned int numDiscLayersB = static_cast<unsigned int>(self.discriminatorB.tensorDff.sizes.size());
					penultActB = ctx.discBFakeAct[numDiscLayersB - 2u];
				}
				else
				{
					std::vector<float> discOutTmp;
					std::vector<std::vector<float> > cnnFcAct;
					self.cnnForward(self.discriminatorB, &fakeB[0], discOutTmp, discSig, &cnnFcAct);
					const unsigned int numCnnFC = static_cast<unsigned int>(cnnFcAct.size());
					penultActB = (numCnnFC >= 2u) ? cnnFcAct[numCnnFC - 2u] : cnnFcAct[0];
				}

				self.qHeadForward(self.qHeadB, &penultActB[0], d.penultDimB, ctx.qOut);
				float miLoss = self.computeInfoLoss(ctx.qOut, d.catCodesAB[s], d.contCodesAB[s], ctx.qGrad);
				ctx.infoLoss += miLoss;
				for (unsigned int i = 0; i < ctx.qGrad.size(); ++i) ctx.qGrad[i] *= d.infoLambda;
				self.qHeadBackward(self.qHeadB, &penultActB[0], &ctx.qGrad[0], ctx.sharedGrad, ctx.qBGrads);

				// Backprop Q gradient through disc B to input
				if (self.config.archType == GANConfig::GAN_DFF)
				{
					const unsigned int numDiscLayersB = static_cast<unsigned int>(self.discriminatorB.tensorDff.sizes.size());
					const unsigned int lastT = static_cast<unsigned int>(self.discriminatorB.tensorDff.T.size());
					if (lastT >= 1u)
					{
						const NNetwork::TensorDFFState& dff = self.discriminatorB.tensorDff;
						const NNInfo* skel = self.discriminatorB.skeleton;
						ctx.qDelta.resize(numDiscLayersB);
						for (unsigned int li = 0; li < numDiscLayersB; ++li)
							ctx.qDelta[li].assign(dff.sizes[li], 0.0f);
						for (unsigned int i = 0; i < d.penultDimB && i < ctx.sharedGrad.size(); ++i)
						{
							if (numDiscLayersB >= 3u)
							{
								const unsigned int tIdx = numDiscLayersB - 3u;
								const int actFx = skel->getActivationType(tIdx);
								const float actParam = skel->getActivationParam(tIdx);
								ctx.qDelta[numDiscLayersB - 2u][i] = ctx.sharedGrad[i] * GMath::activationErrDer(ctx.discBFakeAct[numDiscLayersB - 2u][i], actFx, actParam);
							}
							else
								ctx.qDelta[numDiscLayersB - 2u][i] = ctx.sharedGrad[i];
						}
						for (int li = static_cast<int>(numDiscLayersB) - 3; li >= 0; --li)
						{
							const unsigned int l = static_cast<unsigned int>(li);
							const NNetwork::TensorDFFState::Transition& nextTr = dff.T[l];
							for (unsigned int i = 0; i < dff.sizes[l]; ++i)
							{
								float sum = 0.0f;
								for (unsigned int j = 0; j < nextTr.out; ++j)
									sum += ctx.qDelta[l + 1u][j] * nextTr.W[static_cast<size_t>(j) * nextTr.in + i];
								if (l == 0u) ctx.qDelta[0][i] = sum;
								else
								{
									const int actFx = skel->getActivationType(l - 1u);
									const float actParam = skel->getActivationParam(l - 1u);
									ctx.qDelta[l][i] = sum * GMath::activationErrDer(ctx.discBFakeAct[l][i], actFx, actParam);
								}
							}
						}
						dFakeB_info.resize(d.dimB, 0.0f);
						for (unsigned int i = 0; i < d.dimB && i < ctx.qDelta[0].size(); ++i)
							dFakeB_info[i] = ctx.qDelta[0][i];
					}
				}
				else
				{
					std::vector<float> qInputGrad;
					float zeroGrad = 0.0f;
					ctx.discardDiscBGrads.zero();
					self.cnnBackward(self.discriminatorB, &fakeB[0], &zeroGrad, 1u, &qInputGrad,
					                 ctx.discardDiscBGrads, &ctx.sharedGrad);
					dFakeB_info.resize(d.dimB, 0.0f);
					for (unsigned int i = 0; i < d.dimB && i < qInputGrad.size(); ++i)
						dFakeB_info[i] = qInputGrad[i];
				}
			}

			// Q-head A on D_A features from fakeA
			{
				std::vector<float> penultActA;
				if (self.config.archType == GANConfig::GAN_DFF)
				{
					const unsigned int numDiscLayersA = static_cast<unsigned int>(self.discriminator.tensorDff.sizes.size());
					penultActA = ctx.discAFakeAct[numDiscLayersA - 2u];
				}
				else
				{
					std::vector<float> discOutTmp;
					std::vector<std::vector<float> > cnnFcAct;
					self.cnnForward(self.discriminator, &fakeA[0], discOutTmp, discSig, &cnnFcAct);
					const unsigned int numCnnFC = static_cast<unsigned int>(cnnFcAct.size());
					penultActA = (numCnnFC >= 2u) ? cnnFcAct[numCnnFC - 2u] : cnnFcAct[0];
				}

				self.qHeadForward(self.qHead, &penultActA[0], d.penultDimA, ctx.qOut);
				float miLoss = self.computeInfoLoss(ctx.qOut, d.catCodesBA[s], d.contCodesBA[s], ctx.qGrad);
				ctx.infoLoss += miLoss;
				for (unsigned int i = 0; i < ctx.qGrad.size(); ++i) ctx.qGrad[i] *= d.infoLambda;
				self.qHeadBackward(self.qHead, &penultActA[0], &ctx.qGrad[0], ctx.sharedGrad, ctx.qGrads);

				// Backprop Q gradient through disc A to input
				if (self.config.archType == GANConfig::GAN_DFF)
				{
					const unsigned int numDiscLayersA = static_cast<unsigned int>(self.discriminator.tensorDff.sizes.size());
					const unsigned int lastT = static_cast<unsigned int>(self.discriminator.tensorDff.T.size());
					if (lastT >= 1u)
					{
						const NNetwork::TensorDFFState& dff = self.discriminator.tensorDff;
						const NNInfo* skel = self.discriminator.skeleton;
						ctx.qDelta.resize(numDiscLayersA);
						for (unsigned int li = 0; li < numDiscLayersA; ++li)
							ctx.qDelta[li].assign(dff.sizes[li], 0.0f);
						for (unsigned int i = 0; i < d.penultDimA && i < ctx.sharedGrad.size(); ++i)
						{
							if (numDiscLayersA >= 3u)
							{
								const unsigned int tIdx = numDiscLayersA - 3u;
								const int actFx = skel->getActivationType(tIdx);
								const float actParam = skel->getActivationParam(tIdx);
								ctx.qDelta[numDiscLayersA - 2u][i] = ctx.sharedGrad[i] * GMath::activationErrDer(ctx.discAFakeAct[numDiscLayersA - 2u][i], actFx, actParam);
							}
							else
								ctx.qDelta[numDiscLayersA - 2u][i] = ctx.sharedGrad[i];
						}
						for (int li = static_cast<int>(numDiscLayersA) - 3; li >= 0; --li)
						{
							const unsigned int l = static_cast<unsigned int>(li);
							const NNetwork::TensorDFFState::Transition& nextTr = dff.T[l];
							for (unsigned int i = 0; i < dff.sizes[l]; ++i)
							{
								float sum = 0.0f;
								for (unsigned int j = 0; j < nextTr.out; ++j)
									sum += ctx.qDelta[l + 1u][j] * nextTr.W[static_cast<size_t>(j) * nextTr.in + i];
								if (l == 0u) ctx.qDelta[0][i] = sum;
								else
								{
									const int actFx = skel->getActivationType(l - 1u);
									const float actParam = skel->getActivationParam(l - 1u);
									ctx.qDelta[l][i] = sum * GMath::activationErrDer(ctx.discAFakeAct[l][i], actFx, actParam);
								}
							}
						}
						dFakeA_info.resize(d.dimA, 0.0f);
						for (unsigned int i = 0; i < d.dimA && i < ctx.qDelta[0].size(); ++i)
							dFakeA_info[i] = ctx.qDelta[0][i];
					}
				}
				else
				{
					std::vector<float> qInputGrad;
					float zeroGrad = 0.0f;
					ctx.discardDiscGrads.zero();
					self.cnnBackward(self.discriminator, &fakeA[0], &zeroGrad, 1u, &qInputGrad,
					                 ctx.discardDiscGrads, &ctx.sharedGrad);
					dFakeA_info.resize(d.dimA, 0.0f);
					for (unsigned int i = 0; i < d.dimA && i < qInputGrad.size(); ++i)
						dFakeA_info[i] = qInputGrad[i];
				}
			}
		}

		// === Backprop G_AB ===
		{
			// Disc B backward -> dFakeB (disc grads discarded)
			ctx.discardDiscBGrads.zero();
			std::vector<float> dFakeB_adv;
			if (self.config.archType == GANConfig::GAN_DFF)
			{
				self.dffBackward(self.discriminatorB, ctx.discBFakeAct, &dLdFakeForGenB, 1u, &dFakeB_adv,
				                 ctx.discardDiscBGrads, ctx.scratchDeltaLocal);
			}
			else
			{
				self.cnnBackward(self.discriminatorB, &fakeB[0], &dLdFakeForGenB, 1u, &dFakeB_adv,
				                 ctx.discardDiscBGrads, NULL, false);
			}

			// Add Info gradient
			if (d.hasInfo && !dFakeB_info.empty())
				for (unsigned int i = 0; i < d.dimB; ++i)
					dFakeB_adv[i] += dFakeB_info[i];

			// Cycle B: rec_B = G_AB(fake_A [| codes])
			std::vector<float> dRecB(d.dimB);
			l1LossGrad(&recB[0], realB, d.dimB, &dRecB[0]);
			for (unsigned int i = 0; i < d.dimB; ++i)
				dRecB[i] *= d.cycleLambda;
			if (d.genIsDeconv)
				self.deconvBackward(self.generator, ctx.deconvScratchRecB, &dRecB[0], d.dimB, NULL,
				                    ctx.genGrads);
			else
			{
				self.dffBackward(self.generator, ctx.recBAct, &dRecB[0], d.dimB, NULL,
				                 ctx.genGrads, ctx.scratchDeltaLocal,
				                 false); // straight-through
			}

			// Cycle A: rec_A = G_BA(fake_B [| codes]) -> dFakeB_cycle
			std::vector<float> dRecA(d.dimA);
			l1LossGrad(&recA[0], realA, d.dimA, &dRecA[0]);
			for (unsigned int i = 0; i < d.dimA; ++i)
				dRecA[i] *= d.cycleLambda;
			std::vector<float> dFakeB_cycle;
			if (d.genIsDeconv)
				self.deconvBackward(self.generatorBA, ctx.deconvScratchRecA, &dRecA[0], d.dimA, &dFakeB_cycle,
				                    ctx.genBAGrads);
			else
				self.dffBackward(self.generatorBA, ctx.recAAct, &dRecA[0], d.dimA, &dFakeB_cycle,
				                 ctx.genBAGrads, ctx.scratchDeltaLocal,
				                 false); // straight-through

			// Identity: ident_B = G_AB(y_B [| codes])
			std::vector<float> dIdentB(d.dimB);
			l1LossGrad(&identB[0], realB, d.dimB, &dIdentB[0]);
			for (unsigned int i = 0; i < d.dimB; ++i)
				dIdentB[i] *= d.identityLambda;
			if (d.genIsDeconv)
				self.deconvBackward(self.generator, ctx.deconvScratchIdentB, &dIdentB[0], d.dimB, NULL,
				                    ctx.genGrads);
			else
				self.dffBackward(self.generator, ctx.identBAct, &dIdentB[0], d.dimB, NULL,
				                 ctx.genGrads, ctx.scratchDeltaLocal,
				                 false); // straight-through

			// Combined dFakeB
			for (unsigned int i = 0; i < d.dimB; ++i)
				dFakeB_adv[i] += dFakeB_cycle[i];

			// G_AB backward (styled or plain)
			if (d.genIsDeconv)
			{
				self.deconvBackward(self.generator, ctx.deconvScratch, &dFakeB_adv[0], d.dimB, NULL,
				                    ctx.genGrads);
			}
			else if (d.hasStyle)
			{
				self.dffBackwardStyled(self.generator, ctx.genAct, &dFakeB_adv[0], d.dimB,
				                       ctx.wVec, ctx.sXNorms, ctx.sMeans, ctx.sInvStds, ctx.sNoiseVecs,
				                       self.styleAffines, ctx.gNoiseScalesLocal, ctx.dWVec,
				                       ctx.genGrads, ctx.styleGW, ctx.styleGBias, ctx.scratchDeltaLocal);
				self.dffBackward(self.mappingNet, ctx.mapAct, &ctx.dWVec[0],
				                 static_cast<unsigned int>(ctx.dWVec.size()), NULL,
				                 ctx.mappingGrads, ctx.scratchDeltaLocal);
			}
			else if (d.hasLN)
			{
				self.dffBackward(self.generator, ctx.genAct, &dFakeB_adv[0], d.dimB, NULL,
				                 ctx.genGrads, ctx.scratchDeltaLocal,
				                 false, // straight-through
				                 &d.lnScratchAB[tid], ctx.lnGGamma, ctx.lnGBeta);
			}
			else
			{
				self.dffBackward(self.generator, ctx.genAct, &dFakeB_adv[0], d.dimB, NULL,
				                 ctx.genGrads, ctx.scratchDeltaLocal,
				                 false); // straight-through
			}
		}

		// === Backprop G_BA ===
		{
			// Disc A backward -> dFakeA (disc grads discarded)
			ctx.discardDiscGrads.zero();
			std::vector<float> dFakeA_adv;
			if (self.config.archType == GANConfig::GAN_DFF)
			{
				self.dffBackward(self.discriminator, ctx.discAFakeAct, &dLdFakeForGenA, 1u, &dFakeA_adv,
				                 ctx.discardDiscGrads, ctx.scratchDeltaLocal);
			}
			else
			{
				self.cnnBackward(self.discriminator, &fakeA[0], &dLdFakeForGenA, 1u, &dFakeA_adv,
				                 ctx.discardDiscGrads, NULL, false);
			}

			// Add Info gradient
			if (d.hasInfo && !dFakeA_info.empty())
				for (unsigned int i = 0; i < d.dimA; ++i)
					dFakeA_adv[i] += dFakeA_info[i];

			// Cycle B: rec_B = G_AB(fake_A [| codes]) -> dFakeA_cycle
			// Only need the input gradient here; parameter gradients for G_AB
			// were already accumulated in the G_AB section above, so use a
			// discard buffer to avoid double-counting cycle-B param grads.
			std::vector<float> dRecB2(d.dimB);
			l1LossGrad(&recB[0], realB, d.dimB, &dRecB2[0]);
			for (unsigned int i = 0; i < d.dimB; ++i)
				dRecB2[i] *= d.cycleLambda;
			std::vector<float> dFakeA_cycle;
			ctx.discardGenGrads.zero();
			if (d.genIsDeconv)
			{
				self.deconvBackward(self.generator, ctx.deconvScratchRecB, &dRecB2[0], d.dimB, &dFakeA_cycle,
				                    ctx.discardGenGrads);
			}
			else
			{
				self.dffBackward(self.generator, ctx.recBAct, &dRecB2[0], d.dimB, &dFakeA_cycle,
				                 ctx.discardGenGrads, ctx.scratchDeltaLocal,
				                 false); // straight-through
			}

			// Identity: ident_A = G_BA(x_A [| codes])
			std::vector<float> dIdentA(d.dimA);
			l1LossGrad(&identA[0], realA, d.dimA, &dIdentA[0]);
			for (unsigned int i = 0; i < d.dimA; ++i)
				dIdentA[i] *= d.identityLambda;
			if (d.genIsDeconv)
				self.deconvBackward(self.generatorBA, ctx.deconvScratchIdentA, &dIdentA[0], d.dimA, NULL,
				                    ctx.genBAGrads);
			else
				self.dffBackward(self.generatorBA, ctx.identAAct, &dIdentA[0], d.dimA, NULL,
				                 ctx.genBAGrads, ctx.scratchDeltaLocal,
				                 false); // straight-through

			// Combined dFakeA
			for (unsigned int i = 0; i < d.dimA; ++i)
				dFakeA_adv[i] += dFakeA_cycle[i];

			// G_BA backward (styled or plain)
			if (d.genIsDeconv)
			{
				self.deconvBackward(self.generatorBA, ctx.deconvScratchBA, &dFakeA_adv[0], d.dimA, NULL,
				                    ctx.genBAGrads);
			}
			else if (d.hasStyle)
			{
				self.dffBackwardStyled(self.generatorBA, ctx.genBAAct, &dFakeA_adv[0], d.dimA,
				                       ctx.wBAVec, ctx.sBAXNorms, ctx.sBAMeans, ctx.sBAInvStds, ctx.sBANoiseVecs,
				                       self.styleAffinesBA, ctx.gNoiseScalesBALocal, ctx.dWVec,
				                       ctx.genBAGrads, ctx.styleBAGW, ctx.styleBAGBias, ctx.scratchDeltaLocal);
				self.dffBackward(self.mappingNetBA, ctx.mapBAAct, &ctx.dWVec[0],
				                 static_cast<unsigned int>(ctx.dWVec.size()), NULL,
				                 ctx.mappingBAGrads, ctx.scratchDeltaLocal);
			}
			else if (d.hasBALN)
			{
				self.dffBackward(self.generatorBA, ctx.genBAAct, &dFakeA_adv[0], d.dimA, NULL,
				                 ctx.genBAGrads, ctx.scratchDeltaLocal,
				                 false, // straight-through
				                 &d.lnScratchBA[tid], ctx.lnBAGGamma, ctx.lnBAGBeta);
			}
			else
			{
				self.dffBackward(self.generatorBA, ctx.genBAAct, &dFakeA_adv[0], d.dimA, NULL,
				                 ctx.genBAGrads, ctx.scratchDeltaLocal,
				                 false); // straight-through
			}
		}
	}
}

// ============================================================
// Unified Single-Domain Training (standard, info, style, info+style)
// ============================================================

NNetworkStatus GAN::trainSingleDomain(const DataInput* realData, IGANCallbacks* cb)
{
	const bool hasInfo = config.useInfo || (config.variantType == GANConfig::GAN_INFO);
	const bool hasStyle = config.useStyle || (config.variantType == GANConfig::GAN_STYLE);

	const unsigned int trainSize = realData->getTrainSize();
	if (trainSize == 0u)
		return NNetworkStatus(NNetworkStatus::EMPTY_DATA, "GAN::train: no training data");

	const float* firstRow = NULL;
	unsigned int featureCount = 0u;
	realData->getTrainRowView(0, firstRow, featureCount);
	if (featureCount == 0u)
		return NNetworkStatus(NNetworkStatus::EMPTY_DATA, "GAN::train: zero feature count");

	const unsigned int dataDim = featureCount;
	const unsigned int numCat = hasInfo ? config.infoConfig.numCategorical : 0u;
	const unsigned int numCont = hasInfo ? config.infoConfig.numContinuous : 0u;
	const unsigned int genInputDim = config.noiseDim + numCat + numCont;
	const float infoLambda = config.infoConfig.infoLambda;
	const bool genIsDeconv = (config.archType == GANConfig::GAN_CNN) && !config.generatorDeconv.layers.empty();

	// Initialize generator and discriminator tensors
	if (config.archType == GANConfig::GAN_DFF)
	{
		if (!initDFFTensors(generator, genInputDim, dataDim))
			return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init generator");
		if (!initDFFTensors(discriminator, dataDim, 1u))
			return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init discriminator");
	}
	else
	{
		if (genIsDeconv)
		{
			if (!initDeconvTensors(generator, genInputDim, config.generatorDeconv))
				return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init generator deconv");
		}
		else
		{
			if (!initDFFTensors(generator, genInputDim, dataDim))
				return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init generator");
		}
		if (!initCNNTensors(discriminator, config.discriminatorCNN, 1u))
			return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init discriminator CNN");
	}

	initAdamState(genAdamState, generator);
	initAdamState(discAdamState, discriminator);
	if (config.generatorLayerNorm)
		initLayerNorm(genLNParams, generator);

	// Style init
	unsigned int wDim = 0u;
	if (hasStyle)
	{
		wDim = config.styleConfig.mappingWidth;
		if (!initDFFTensors(mappingNet, config.noiseDim, wDim))
			return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init mapping net");
		initAdamState(mappingAdamState, mappingNet);
		initStyleAffines(wDim, generator, styleAffines, styleAffineAdamState, noiseScales, gNoiseScales);
	}

	// Info init
	unsigned int penultDim = 0u;
	if (hasInfo)
	{
		if (config.archType == GANConfig::GAN_DFF)
		{
			const unsigned int numDiscLayers = static_cast<unsigned int>(discriminator.tensorDff.sizes.size());
			penultDim = (numDiscLayers >= 2u) ? discriminator.tensorDff.sizes[numDiscLayers - 2u] : dataDim;
		}
		else
		{
			// CNN: penultimate FC layer output size
			const unsigned int numCnnFC = static_cast<unsigned int>(discriminator.tensorCnn.fcLayers.size());
			penultDim = (numCnnFC >= 2u) ? discriminator.tensorCnn.fcLayers[numCnnFC - 2u].out : discriminator.tensorCnn.flattenedSize;
		}
		initQHead(penultDim, qHead, qAdamState);
		// genQHead no longer used — info gradient flows through disc-side qHead
	}

	const int batchSize = (config.batchSize > 0) ? config.batchSize : static_cast<int>(trainSize);
	const unsigned int numBatches = (trainSize + static_cast<unsigned int>(batchSize) - 1u) / static_cast<unsigned int>(batchSize);

	// --- Pre-allocated scratch buffers (reused across all samples) ---
	std::vector<float> noise;
	std::vector<float> catCode, contCode;
	std::vector<float> genInput(genInputDim);
	// Styled path scratch
	std::vector<std::vector<float> > mapAct;
	std::vector<std::vector<float> > genAct, xN;
	std::vector<float> gM, gI;
	std::vector<std::vector<float> > gNV;
	std::vector<float> wVec;
	std::vector<std::vector<float> > sXNorms;
	std::vector<float> sMeans, sInvStds;
	std::vector<std::vector<float> > sNoiseVecs;
	// Discriminator scratch
	std::vector<std::vector<float> > discActReal, discActFake;
	std::vector<float> cnnOutReal, cnnOutFake;
	// Backward scratch
	std::vector<float> dFake;
	std::vector<float> dWVec;
	// Deconv generator scratch
	std::vector<float> deconvOut;
	std::vector<std::vector<float> > deconvScratch;
	// InfoGAN scratch
	std::vector<float> qOut, qGrad, sharedGrad;
	std::vector<std::vector<float> > qDelta;

	// --- Parallel training setup ---
	const unsigned int nThreads = ThreadPool::instance().numThreads();
	std::vector<GANThreadCtx> threadCtx(nThreads);
	for (unsigned int tid = 0; tid < nThreads; ++tid)
	{
		threadCtx[tid].genInput.resize(genInputDim, 0.0f);
		if (genIsDeconv)
		{
			threadCtx[tid].genGrads.initFromDeconv(generator);
			threadCtx[tid].deconvArena.initFromDeconv(generator);
		}
		else
			threadCtx[tid].genGrads.initFromDFF(generator);
		if (config.archType == GANConfig::GAN_DFF)
			threadCtx[tid].discGrads.initFromDFF(discriminator);
		else
			threadCtx[tid].discGrads.initFromCNN(discriminator);
		if (hasInfo)
		{
			threadCtx[tid].qGrads.initFromQHead(qHead.sharedDim, qHead.qOutDim);
			threadCtx[tid].genGrads.initFromGenQHead(genQHead.sharedDim, genQHead.qOutDim, genQHead.hiddenDim);
			threadCtx[tid].genQHiddenPre.assign(genQHead.hiddenDim, 0.0f);
			threadCtx[tid].genQHiddenPost.assign(genQHead.hiddenDim, 0.0f);
			threadCtx[tid].genQDHidden.assign(genQHead.hiddenDim, 0.0f);
			threadCtx[tid].penultActBuf.resize(penultDim, 0.0f);
			threadCtx[tid].qInfoDFakeBuf.resize(dataDim, 0.0f);
		}
		if (hasStyle)
		{
			threadCtx[tid].mappingGrads.initFromDFF(mappingNet);
			threadCtx[tid].styleGW.resize(styleAffines.size());
			threadCtx[tid].styleGBias.resize(styleAffines.size());
			for (size_t sa = 0; sa < styleAffines.size(); ++sa)
			{
				threadCtx[tid].styleGW[sa].assign(styleAffines[sa].gW.size(), 0.0f);
				threadCtx[tid].styleGBias[sa].assign(styleAffines[sa].gBias.size(), 0.0f);
			}
			threadCtx[tid].gNoiseScalesLocal.assign(noiseScales.size(), 0.0f);
		}
		// Discard buffer for GP penalty (same arch as discriminator)
		if (config.archType == GANConfig::GAN_DFF)
			threadCtx[tid].discardDiscGrads.initFromDFF(discriminator);
		else
			threadCtx[tid].discardDiscGrads.initFromCNN(discriminator);
		// LayerNorm gradient buffers
		if (config.generatorLayerNorm && !hasStyle && genLNParams.initialized)
		{
			threadCtx[tid].lnGGamma.resize(genLNParams.gGamma.size());
			threadCtx[tid].lnGBeta.resize(genLNParams.gBeta.size());
			for (size_t t = 0; t < genLNParams.gGamma.size(); ++t)
			{
				threadCtx[tid].lnGGamma[t].assign(genLNParams.gGamma[t].size(), 0.0f);
				threadCtx[tid].lnGBeta[t].assign(genLNParams.gBeta[t].size(), 0.0f);
			}
		}
	}

	// Pre-allocation buffers for batch data
	std::vector<const float*> preRealPtrs;
	std::vector<unsigned int> preRealSizes;
	std::vector<std::vector<float> > preNoise;
	std::vector<std::vector<float> > preCatCode, preContCode;
	std::vector<float> preGPEpsilon;
	std::vector<unsigned int> beginToTid;

	// Pre-allocate per-thread LN forward scratch (shared across batches)
	const bool hasLN = config.generatorLayerNorm && !hasStyle && genLNParams.initialized;
	std::vector<LayerNormParams> genLNScratch;
	if (hasLN)
	{
		genLNScratch.resize(nThreads);
		for (unsigned int tid = 0; tid < nThreads; ++tid)
		{
			genLNScratch[tid].gamma = genLNParams.gamma;
			genLNScratch[tid].beta = genLNParams.beta;
			genLNScratch[tid].zNorm.resize(genLNParams.zNorm.size());
			genLNScratch[tid].invStd.resize(genLNParams.invStd.size(), 0.0f);
			for (size_t t = 0; t < genLNParams.zNorm.size(); ++t)
				genLNScratch[tid].zNorm[t].assign(genLNParams.zNorm[t].size(), 0.0f);
			genLNScratch[tid].gGamma.resize(genLNParams.gGamma.size());
			genLNScratch[tid].gBeta.resize(genLNParams.gBeta.size());
			for (size_t t = 0; t < genLNParams.gGamma.size(); ++t)
			{
				genLNScratch[tid].gGamma[t].assign(genLNParams.gGamma[t].size(), 0.0f);
				genLNScratch[tid].gBeta[t].assign(genLNParams.gBeta[t].size(), 0.0f);
			}
			genLNScratch[tid].initialized = true;
		}
	}

	for (int epoch = 0; epoch < config.epochs; ++epoch)
	{
		float epochDLossReal = 0.0f;
		float epochDLossFake = 0.0f;
		float epochGLoss = 0.0f;
		float epochWasserstein = 0.0f;
		float epochInfoLoss = 0.0f;
		float epochDivLoss = 0.0f;
		float epochDOutReal = 0.0f;
		float epochDOutFake = 0.0f;
		float epochGenGradNorm = 0.0f;
		unsigned int epochCatCorrect = 0u;
		unsigned int epochCatTotal = 0u;
		unsigned int epochSamples = 0u;
		unsigned int dataIdx = 0u;

		for (unsigned int batch = 0; batch < numBatches; ++batch)
		{
			const unsigned int curBatchSize =
			    (static_cast<int>(trainSize - dataIdx) < batchSize)
			        ? (trainSize - dataIdx)
			        : static_cast<unsigned int>(batchSize);

			// --- Train Discriminator ---
			for (int critic = 0; critic < config.nCriticPerGenerator; ++critic)
			{
				float batchDLossReal = 0.0f;
				float batchDLossFake = 0.0f;
				float batchInfoLoss = 0.0f;

				// --- Phase 1: Pre-generate data (sequential, preserves RNG determinism) ---
				preRealPtrs.resize(curBatchSize);
				preRealSizes.resize(curBatchSize);
				for (unsigned int s = 0; s < curBatchSize; ++s)
				{
					const unsigned int realIdx = (dataIdx + s) % trainSize;
					realData->getTrainRowView(realIdx, preRealPtrs[s], preRealSizes[s]);
				}
				preNoise.resize(curBatchSize);
				if (hasInfo) { preCatCode.resize(curBatchSize); preContCode.resize(curBatchSize); }
				if (config.lossType == GANConfig::GAN_WGAN_GP) preGPEpsilon.resize(curBatchSize);
				for (unsigned int s = 0; s < curBatchSize; ++s)
				{
					sampleNoise(preNoise[s]);
					if (hasInfo) sampleLatentCodes(preCatCode[s], preContCode[s]);
					if (config.lossType == GANConfig::GAN_WGAN_GP)
						preGPEpsilon[s] = glades::rng::unit_float01(rngEngine);
				}

				// --- Phase 2: Zero thread contexts ---
				for (unsigned int tid = 0; tid < nThreads; ++tid)
				{
					threadCtx[tid].zeroGrads();
					threadCtx[tid].zeroLosses();
				}

				// --- Phase 3: Parallel compute ---
				// Build begin-to-tid mapping (mirrors ThreadPool chunk assignment)
				beginToTid.assign(curBatchSize, 0u);
				for (unsigned int id = 0; id < nThreads; ++id)
				{
					const unsigned int b = id * curBatchSize / nThreads;
					beginToTid[b] = id;
				}

				DiscCritData dcd;
				dcd.self = this;
				dcd.ctxs = &threadCtx[0];
				dcd.nThreads = nThreads;
				dcd.curBatchSize = curBatchSize;
				dcd.realPtrs = &preRealPtrs[0];
				dcd.realSizes = &preRealSizes[0];
				dcd.noise = &preNoise[0];
				dcd.catCodes = hasInfo ? &preCatCode[0] : NULL;
				dcd.contCodes = hasInfo ? &preContCode[0] : NULL;
				dcd.gpEps = (config.lossType == GANConfig::GAN_WGAN_GP) ? &preGPEpsilon[0] : NULL;
				dcd.genInputDim = genInputDim;
				dcd.dataDim = dataDim;
				dcd.numCat = numCat;
				dcd.numCont = numCont;
				dcd.hasInfo = hasInfo;
				dcd.hasStyle = hasStyle;
				dcd.genIsDeconv = genIsDeconv;
				dcd.infoLambda = infoLambda;
				dcd.catScale = config.infoConfig.catScale;
				dcd.penultDim = penultDim;
				dcd.beginToTid = &beginToTid[0];
				ThreadPool::instance().parallel_for(curBatchSize, discCritBody, &dcd);

				// --- Phase 4: Ordered reduction (sum thread-local grads into network) ---
				if (config.deterministicReduce)
				{
					// Zero master network grads
					if (config.archType == GANConfig::GAN_DFF)
						zeroDFFGrads(discriminator);
					else
						zeroCNNGrads(discriminator);
					if (hasInfo)
					{
						std::memset(&qHead.gW[0], 0, qHead.gW.size() * sizeof(float));
						std::memset(&qHead.gBias[0], 0, qHead.gBias.size() * sizeof(float));
					}

					// Sum thread-local grads into master (in order for determinism)
					for (unsigned int tid = 0; tid < nThreads; ++tid)
					{
						if (config.archType == GANConfig::GAN_DFF)
							threadCtx[tid].discGrads.addToDFF(discriminator);
						else
							threadCtx[tid].discGrads.addToCNN(discriminator);
						if (hasInfo)
						{
							for (size_t i = 0; i < qHead.gW.size(); ++i)
								qHead.gW[i] += threadCtx[tid].qGrads.qGW[i];
							for (size_t j = 0; j < qHead.gBias.size(); ++j)
								qHead.gBias[j] += threadCtx[tid].qGrads.qGBias[j];
						}
						batchDLossReal += threadCtx[tid].dLossReal;
						batchDLossFake += threadCtx[tid].dLossFake;
						batchInfoLoss += threadCtx[tid].infoLoss;
						epochWasserstein += threadCtx[tid].wasserstein;
						epochCatCorrect += threadCtx[tid].catCorrect;
						epochCatTotal += threadCtx[tid].catTotal;
						epochDOutReal += threadCtx[tid].dOutRealSum;
						epochDOutFake += threadCtx[tid].dOutFakeSum;
					}
				}
				else
				{
					// Parallel layer-wise reduction
					LayerReduceData lrd;
					lrd.ctxs = &threadCtx[0];
					lrd.nThreads = nThreads;
					lrd.net = &discriminator;
					lrd.isGen = false;
					const unsigned int nConv = static_cast<unsigned int>(discriminator.tensorCnn.convLayers.size());
					const unsigned int nFC = static_cast<unsigned int>(discriminator.tensorCnn.fcLayers.size());
					ThreadPool::instance().parallel_for(nConv + nFC, layerReduceBody, &lrd);

					// Q-head (small, serial)
					if (hasInfo)
					{
						std::memset(&qHead.gW[0], 0, qHead.gW.size() * sizeof(float));
						std::memset(&qHead.gBias[0], 0, qHead.qOutDim * sizeof(float));
						for (unsigned int tid = 0; tid < nThreads; ++tid)
						{
							for (size_t i = 0; i < qHead.gW.size(); ++i)
								qHead.gW[i] += threadCtx[tid].qGrads.qGW[i];
							for (size_t j = 0; j < qHead.qOutDim; ++j)
								qHead.gBias[j] += threadCtx[tid].qGrads.qGBias[j];
						}
					}

					// Loss accumulators (tiny, serial)
					for (unsigned int tid = 0; tid < nThreads; ++tid)
					{
						batchDLossReal += threadCtx[tid].dLossReal;
						batchDLossFake += threadCtx[tid].dLossFake;
						batchInfoLoss += threadCtx[tid].infoLoss;
						epochWasserstein += threadCtx[tid].wasserstein;
						epochCatCorrect += threadCtx[tid].catCorrect;
						epochCatTotal += threadCtx[tid].catTotal;
						epochDOutReal += threadCtx[tid].dOutRealSum;
						epochDOutFake += threadCtx[tid].dOutFakeSum;
					}
				}

				const float invBatch = 1.0f / static_cast<float>(curBatchSize);
				epochDLossReal += batchDLossReal * invBatch;
				epochDLossFake += batchDLossFake * invBatch;
				if (hasInfo) epochInfoLoss += batchInfoLoss * invBatch;

				// Adaptive balance: skip disc weight update when disc is dominant
				const float avgDiscLoss = (batchDLossReal + batchDLossFake) * invBatch;
				const bool skipDiscUpdate = config.adaptiveBalance &&
				    (avgDiscLoss < config.adaptiveDiscThreshold);
				if (!skipDiscUpdate)
				{
					// Scale and update discriminator
					if (config.archType == GANConfig::GAN_DFF)
					{
						for (unsigned int t = 0; t < discriminator.tensorDff.T.size(); ++t)
						{
							NNetwork::TensorDFFState::Transition& tr = discriminator.tensorDff.T[t];
							for (size_t i = 0; i < tr.gW.size(); ++i) tr.gW[i] *= invBatch;
							for (size_t j = 0; j < tr.gBias.size(); ++j) tr.gBias[j] *= invBatch;
						}
						dffUpdate(discriminator, discAdamState, config.discriminatorLR);
					}
					else
					{
						for (unsigned int l = 0; l < discriminator.tensorCnn.convLayers.size(); ++l)
						{
							NNetwork::TensorCNNState::ConvLayer& cl = discriminator.tensorCnn.convLayers[l];
							for (size_t i = 0; i < cl.gW.size(); ++i) cl.gW[i] *= invBatch;
							for (unsigned int c = 0; c < cl.outC; ++c) cl.gBias[c] *= invBatch;
						}
						for (unsigned int t = 0; t < discriminator.tensorCnn.fcLayers.size(); ++t)
						{
							NNetwork::TensorCNNState::FCTransition& fc = discriminator.tensorCnn.fcLayers[t];
							for (size_t i = 0; i < fc.gW.size(); ++i) fc.gW[i] *= invBatch;
							for (unsigned int j = 0; j < fc.out; ++j) fc.gBias[j] *= invBatch;
						}
						cnnUpdate(discriminator, discAdamState, config.discriminatorLR);
					}

					// Spectral normalization: constrain disc Lipschitz constant
					if (config.spectralNorm)
					{
						if (config.archType == GANConfig::GAN_DFF)
							spectralNormDFF(discriminator, discSNState, config.spectralNormIters);
						else
							spectralNormCNN(discriminator, discSNState, config.spectralNormIters);
					}

					// Update Q-head
					if (hasInfo)
					{
						for (size_t i = 0; i < qHead.gW.size(); ++i) qHead.gW[i] *= invBatch;
						for (unsigned int j = 0; j < qHead.qOutDim; ++j) qHead.gBias[j] *= invBatch;
						qHeadUpdate(qHead, qAdamState, config.discriminatorLR);
					}
				}
			}

			// --- Train Generator ---
			for (int gstep = 0; gstep < config.nGenPerCritic; ++gstep)
			{
				float batchGLoss = 0.0f;
				float batchInfoLoss = 0.0f;

				// --- Phase 1: Pre-generate noise and codes (sequential, preserves RNG determinism) ---
				preNoise.resize(curBatchSize);
				if (hasInfo) { preCatCode.resize(curBatchSize); preContCode.resize(curBatchSize); }
				for (unsigned int s = 0; s < curBatchSize; ++s)
				{
					sampleNoise(preNoise[s]);
					if (hasInfo) sampleLatentCodes(preCatCode[s], preContCode[s]);
				}

				// --- Phase 2: Zero thread contexts ---
				for (unsigned int tid = 0; tid < nThreads; ++tid)
				{
					threadCtx[tid].zeroGrads();
					threadCtx[tid].zeroLosses();
				}

				// Refresh per-thread LN scratch gamma/beta (updated by Adam each batch)
				if (hasLN)
				{
					for (unsigned int tid = 0; tid < nThreads; ++tid)
					{
						genLNScratch[tid].gamma = genLNParams.gamma;
						genLNScratch[tid].beta = genLNParams.beta;
					}
				}

				// --- Phase 3: Parallel compute ---
				beginToTid.assign(curBatchSize, 0u);
				for (unsigned int id = 0; id < nThreads; ++id)
				{
					const unsigned int b = id * curBatchSize / nThreads;
					beginToTid[b] = id;
				}

				GenTrainData gtd;
				gtd.self = this;
				gtd.ctxs = &threadCtx[0];
				gtd.nThreads = nThreads;
				gtd.curBatchSize = curBatchSize;
				gtd.noise = &preNoise[0];
				gtd.catCodes = hasInfo ? &preCatCode[0] : NULL;
				gtd.contCodes = hasInfo ? &preContCode[0] : NULL;
				gtd.genInputDim = genInputDim;
				gtd.dataDim = dataDim;
				gtd.numCat = numCat;
				gtd.numCont = numCont;
				gtd.hasInfo = hasInfo;
				gtd.hasStyle = hasStyle;
				gtd.genIsDeconv = genIsDeconv;
				gtd.hasLN = hasLN;
				gtd.infoLambda = infoLambda;
				gtd.catScale = config.infoConfig.catScale;
				gtd.divLambda = config.infoConfig.divLambda;
				gtd.penultDim = penultDim;
				gtd.beginToTid = &beginToTid[0];
				gtd.lnScratch = hasLN ? &genLNScratch[0] : NULL;
				ThreadPool::instance().parallel_for(curBatchSize, genTrainBody, &gtd);

				// --- Phase 4: Ordered reduction ---
				if (config.deterministicReduce)
				{
					// Zero master network grads
					if (genIsDeconv)
						zeroDeconvGrads(generator);
					else
						zeroDFFGrads(generator);
					if (hasStyle && !genIsDeconv)
					{
						zeroDFFGrads(mappingNet);
						for (size_t sa = 0; sa < styleAffines.size(); ++sa)
						{
							std::memset(&styleAffines[sa].gW[0], 0, styleAffines[sa].gW.size() * sizeof(float));
							std::memset(&styleAffines[sa].gBias[0], 0, styleAffines[sa].gBias.size() * sizeof(float));
							gNoiseScales[sa] = 0.0f;
						}
					}
					if (hasLN)
					{
						for (size_t i = 0; i < genLNParams.gGamma.size(); ++i)
						{
							std::memset(&genLNParams.gGamma[i][0], 0, genLNParams.gGamma[i].size() * sizeof(float));
							std::memset(&genLNParams.gBeta[i][0], 0, genLNParams.gBeta[i].size() * sizeof(float));
						}
					}
					// genQHead no longer used — info gradient flows through disc-side qHead

					// Sum thread-local grads into master (in order for determinism)
					for (unsigned int tid = 0; tid < nThreads; ++tid)
					{
						if (genIsDeconv)
							threadCtx[tid].genGrads.addToDeconv(generator);
						else
							threadCtx[tid].genGrads.addToDFF(generator);

						if (hasStyle && !genIsDeconv)
						{
							threadCtx[tid].mappingGrads.addToDFF(mappingNet);
							for (size_t sa = 0; sa < styleAffines.size(); ++sa)
							{
								for (size_t i = 0; i < styleAffines[sa].gW.size(); ++i)
									styleAffines[sa].gW[i] += threadCtx[tid].styleGW[sa][i];
								for (size_t j = 0; j < styleAffines[sa].gBias.size(); ++j)
									styleAffines[sa].gBias[j] += threadCtx[tid].styleGBias[sa][j];
								gNoiseScales[sa] += threadCtx[tid].gNoiseScalesLocal[sa];
							}
						}
						if (hasLN)
						{
							for (size_t i = 0; i < genLNParams.gGamma.size(); ++i)
							{
								for (size_t j = 0; j < genLNParams.gGamma[i].size(); ++j)
									genLNParams.gGamma[i][j] += threadCtx[tid].lnGGamma[i][j];
								for (size_t j = 0; j < genLNParams.gBeta[i].size(); ++j)
									genLNParams.gBeta[i][j] += threadCtx[tid].lnGBeta[i][j];
							}
						}
						// genQHead gradient accumulation removed — Q updated during disc training only

						batchGLoss += threadCtx[tid].gLoss;
						batchInfoLoss += threadCtx[tid].infoLoss;
						epochDivLoss += threadCtx[tid].divLoss;
					}
				}
				else
				{
					// Parallel layer-wise reduction (deconv generator only)
					if (genIsDeconv)
					{
						const unsigned int numDeconvLayers = static_cast<unsigned int>(generator.tensorDeconv.layers.size());
						LayerReduceData lrd;
						lrd.ctxs = &threadCtx[0];
						lrd.nThreads = nThreads;
						lrd.net = &generator;
						lrd.isGen = true;
						ThreadPool::instance().parallel_for(1u + numDeconvLayers, layerReduceBody, &lrd);
					}
					else
					{
						zeroDFFGrads(generator);
						for (unsigned int tid = 0; tid < nThreads; ++tid)
							threadCtx[tid].genGrads.addToDFF(generator);
					}

					// Style / mapping reduction (serial — small and complex)
					if (hasStyle && !genIsDeconv)
					{
						zeroDFFGrads(mappingNet);
						for (size_t sa = 0; sa < styleAffines.size(); ++sa)
						{
							std::memset(&styleAffines[sa].gW[0], 0, styleAffines[sa].gW.size() * sizeof(float));
							std::memset(&styleAffines[sa].gBias[0], 0, styleAffines[sa].gBias.size() * sizeof(float));
							gNoiseScales[sa] = 0.0f;
						}
						for (unsigned int tid = 0; tid < nThreads; ++tid)
						{
							threadCtx[tid].mappingGrads.addToDFF(mappingNet);
							for (size_t sa = 0; sa < styleAffines.size(); ++sa)
							{
								for (size_t i = 0; i < styleAffines[sa].gW.size(); ++i)
									styleAffines[sa].gW[i] += threadCtx[tid].styleGW[sa][i];
								for (size_t j = 0; j < styleAffines[sa].gBias.size(); ++j)
									styleAffines[sa].gBias[j] += threadCtx[tid].styleGBias[sa][j];
								gNoiseScales[sa] += threadCtx[tid].gNoiseScalesLocal[sa];
							}
						}
					}

					// LN gradient reduction (serial — small and complex)
					if (hasLN)
					{
						for (size_t i = 0; i < genLNParams.gGamma.size(); ++i)
						{
							std::memset(&genLNParams.gGamma[i][0], 0, genLNParams.gGamma[i].size() * sizeof(float));
							std::memset(&genLNParams.gBeta[i][0], 0, genLNParams.gBeta[i].size() * sizeof(float));
						}
						for (unsigned int tid = 0; tid < nThreads; ++tid)
						{
							for (size_t i = 0; i < genLNParams.gGamma.size(); ++i)
							{
								for (size_t j = 0; j < genLNParams.gGamma[i].size(); ++j)
									genLNParams.gGamma[i][j] += threadCtx[tid].lnGGamma[i][j];
								for (size_t j = 0; j < genLNParams.gBeta[i].size(); ++j)
									genLNParams.gBeta[i][j] += threadCtx[tid].lnGBeta[i][j];
							}
						}
					}

					// Loss accumulators (tiny, serial)
					for (unsigned int tid = 0; tid < nThreads; ++tid)
					{
						batchGLoss += threadCtx[tid].gLoss;
						batchInfoLoss += threadCtx[tid].infoLoss;
						epochDivLoss += threadCtx[tid].divLoss;
					}
				}

				const float invBatch = 1.0f / static_cast<float>(curBatchSize);
				epochGLoss += batchGLoss * invBatch;

				// Scale and update generator
				if (genIsDeconv)
				{
					// Scale deconv gradients
					NNetwork::TensorDeconvState& ds = generator.tensorDeconv;
					for (size_t i = 0; i < ds.fcGW.size(); ++i) ds.fcGW[i] *= invBatch;
					for (unsigned int j = 0; j < ds.fcOut; ++j) ds.fcGBias[j] *= invBatch;
					for (unsigned int l = 0; l < ds.layers.size(); ++l)
					{
						NNetwork::TensorDeconvState::DeconvLayer& dl = ds.layers[l];
						for (size_t i = 0; i < dl.gW.size(); ++i) dl.gW[i] *= invBatch;
						for (unsigned int c = 0; c < dl.outC; ++c) dl.gBias[c] *= invBatch;
						if (dl.useBatchNorm)
						{
							for (unsigned int c = 0; c < dl.outC; ++c) dl.gBnGamma[c] *= invBatch;
							for (unsigned int c = 0; c < dl.outC; ++c) dl.gBnBeta[c] *= invBatch;
						}
					}

					// Generator gradient L2 norm (after batch-averaging)
					float gnNorm = 0.0f;
					{
						float gnSq = 0.0f;
						for (size_t i = 0; i < ds.fcGW.size(); ++i) gnSq += ds.fcGW[i] * ds.fcGW[i];
						for (unsigned int j = 0; j < ds.fcOut; ++j) gnSq += ds.fcGBias[j] * ds.fcGBias[j];
						for (unsigned int l2 = 0; l2 < ds.layers.size(); ++l2)
						{
							const NNetwork::TensorDeconvState::DeconvLayer& dl2 = ds.layers[l2];
							for (size_t i = 0; i < dl2.gW.size(); ++i) gnSq += dl2.gW[i] * dl2.gW[i];
							for (unsigned int c = 0; c < dl2.outC; ++c) gnSq += dl2.gBias[c] * dl2.gBias[c];
						}
						gnNorm = sqrtf(gnSq);
						epochGenGradNorm += gnNorm;
					}

					// Gradient clipping (deconv generator)
					if (config.gradClipNorm > 0.0f && gnNorm > config.gradClipNorm)
					{
						const float scale = config.gradClipNorm / (gnNorm + 1e-8f);
						for (size_t i = 0; i < ds.fcGW.size(); ++i) ds.fcGW[i] *= scale;
						for (unsigned int j = 0; j < ds.fcOut; ++j) ds.fcGBias[j] *= scale;
						for (unsigned int l2 = 0; l2 < ds.layers.size(); ++l2)
						{
							NNetwork::TensorDeconvState::DeconvLayer& dl2 = ds.layers[l2];
							for (size_t i = 0; i < dl2.gW.size(); ++i) dl2.gW[i] *= scale;
							for (unsigned int c = 0; c < dl2.outC; ++c) dl2.gBias[c] *= scale;
							if (dl2.useBatchNorm)
							{
								for (unsigned int c = 0; c < dl2.outC; ++c) dl2.gBnGamma[c] *= scale;
								for (unsigned int c = 0; c < dl2.outC; ++c) dl2.gBnBeta[c] *= scale;
							}
						}
					}

					deconvUpdate(generator, config.generatorLR);
				}
				else
				{
					for (unsigned int t = 0; t < generator.tensorDff.T.size(); ++t)
					{
						NNetwork::TensorDFFState::Transition& tr = generator.tensorDff.T[t];
						for (size_t i = 0; i < tr.gW.size(); ++i) tr.gW[i] *= invBatch;
						for (size_t j = 0; j < tr.gBias.size(); ++j) tr.gBias[j] *= invBatch;
					}

					// Generator gradient L2 norm (after batch-averaging)
					float gnNormDff = 0.0f;
					{
						float gnSq = 0.0f;
						for (unsigned int t2 = 0; t2 < generator.tensorDff.T.size(); ++t2)
						{
							const NNetwork::TensorDFFState::Transition& tr2 = generator.tensorDff.T[t2];
							for (size_t i = 0; i < tr2.gW.size(); ++i) gnSq += tr2.gW[i] * tr2.gW[i];
							for (size_t j = 0; j < tr2.gBias.size(); ++j) gnSq += tr2.gBias[j] * tr2.gBias[j];
						}
						gnNormDff = sqrtf(gnSq);
						epochGenGradNorm += gnNormDff;
					}

					// Gradient clipping (DFF generator)
					if (config.gradClipNorm > 0.0f && gnNormDff > config.gradClipNorm)
					{
						const float scale = config.gradClipNorm / (gnNormDff + 1e-8f);
						for (unsigned int t2 = 0; t2 < generator.tensorDff.T.size(); ++t2)
						{
							NNetwork::TensorDFFState::Transition& tr2 = generator.tensorDff.T[t2];
							for (size_t i = 0; i < tr2.gW.size(); ++i) tr2.gW[i] *= scale;
							for (size_t j = 0; j < tr2.gBias.size(); ++j) tr2.gBias[j] *= scale;
						}
					}

					dffUpdate(generator, genAdamState, config.generatorLR);
					if (hasLN)
					{
						scaleLayerNormGrads(genLNParams, invBatch);
						layerNormUpdate(genLNParams, config.generatorLR);
					}
				}

				if (hasStyle && !genIsDeconv)
				{
					// Scale and update mapping
					for (unsigned int t = 0; t < mappingNet.tensorDff.T.size(); ++t)
					{
						NNetwork::TensorDFFState::Transition& tr = mappingNet.tensorDff.T[t];
						for (size_t i = 0; i < tr.gW.size(); ++i) tr.gW[i] *= invBatch;
						for (size_t j = 0; j < tr.gBias.size(); ++j) tr.gBias[j] *= invBatch;
					}
					dffUpdate(mappingNet, mappingAdamState, config.generatorLR);

					// Scale and update style affines
					for (unsigned int t = 0; t < styleAffines.size(); ++t)
					{
						for (size_t i = 0; i < styleAffines[t].gW.size(); ++i) styleAffines[t].gW[i] *= invBatch;
						for (unsigned int j = 0; j < styleAffines[t].outDim; ++j) styleAffines[t].gBias[j] *= invBatch;
						gNoiseScales[t] *= invBatch;
					}
					styleAffineUpdate(config.generatorLR, styleAffines, styleAffineAdamState, noiseScales, gNoiseScales);
				}

				// genQHead update removed — Q updated during disc training only
			}

			epochSamples += curBatchSize;
			dataIdx += curBatchSize;
		}

		const float invBatches = 1.0f / static_cast<float>(numBatches);
		GANEpochMetrics metrics;
		metrics.epoch = epoch;
		metrics.dLossReal = epochDLossReal * invBatches;
		metrics.dLossFake = epochDLossFake * invBatches;
		metrics.gLoss = epochGLoss * invBatches;
		if (config.lossType == GANConfig::GAN_WGAN_GP && epochSamples > 0u)
			metrics.wasserstein = epochWasserstein / static_cast<float>(epochSamples);
		if (hasInfo)
		{
			metrics.infoLoss = epochInfoLoss * invBatches;
			metrics.divLoss = epochDivLoss * invBatches;
			if (epochCatTotal > 0u)
				metrics.catAccuracy = static_cast<float>(epochCatCorrect) / static_cast<float>(epochCatTotal);
		}

		// Health diagnostics
		if (epochSamples > 0u)
		{
			const float invSamples = 1.0f / static_cast<float>(epochSamples);
			metrics.dOutReal = epochDOutReal * invSamples;
			metrics.dOutFake = epochDOutFake * invSamples;
		}
		const unsigned int genStepsPerEpoch = numBatches * static_cast<unsigned int>(config.nGenPerCritic);
		if (genStepsPerEpoch > 0u)
			metrics.genGradNorm = epochGenGradNorm / static_cast<float>(genStepsPerEpoch);

		// Sample diversity: mean per-pixel variance across generated samples
		{
			const unsigned int divSamples = 16u;
			std::vector<std::vector<float> > divBatch;
			generate(divSamples, divBatch);
			if (divBatch.size() == divSamples && divBatch[0].size() > 0u)
			{
				const unsigned int dim = static_cast<unsigned int>(divBatch[0].size());
				float totalVar = 0.0f;
				for (unsigned int d = 0; d < dim; ++d)
				{
					float sum = 0.0f, sumSq = 0.0f;
					for (unsigned int s = 0; s < divSamples; ++s)
					{
						const float v = divBatch[s][d];
						sum += v;
						sumSq += v * v;
					}
					const float mean = sum / static_cast<float>(divSamples);
					totalVar += sumSq / static_cast<float>(divSamples) - mean * mean;
				}
				metrics.sampleDiversity = totalVar / static_cast<float>(dim);
			}
		}

		if (cb)
		{
			cb->onEpochEnd(metrics);
			if (cb->shouldStop(metrics))
				break;
		}
	}

	return NNetworkStatus();
}

// ============================================================
// Generation
// ============================================================

NNetworkStatus GAN::generate(unsigned int numSamples,
                             std::vector<std::vector<float> >& outSamples) const
{
	const bool isDeconv = generator.tensorDeconv.initialized;
	if (!generator.tensorDff.initialized && !isDeconv)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "GAN::generate: generator not trained");

	const bool hasInfo = config.useInfo || (config.variantType == GANConfig::GAN_INFO);
	const bool hasStyle = config.useStyle || (config.variantType == GANConfig::GAN_STYLE);

	if (hasInfo)
	{
		// Generate with random codes (handles Info+Style combo too)
		return generateInfoGAN(numSamples, NULL, NULL, outSamples);
	}

	if (hasStyle && !isDeconv)
	{
		if (!mappingNet.tensorDff.initialized)
			return NNetworkStatus(NNetworkStatus::INVALID_STATE, "GAN::generate: mapping net not trained");

		outSamples.resize(numSamples);
		for (unsigned int s = 0; s < numSamples; ++s)
		{
			std::vector<float> noise;
			sampleNoise(noise);

			std::vector<std::vector<float> > mapAct;
			dffForward(mappingNet, &noise[0], config.noiseDim, mapAct);
			const std::vector<float>& w = mapAct.back();

			std::vector<std::vector<float> > genAct;
			std::vector<std::vector<float> > xNorms;
			std::vector<float> means, invStds;
			std::vector<std::vector<float> > noiseVecs;
			dffForwardStyled(generator, &noise[0], config.noiseDim, w,
			                 styleAffines, noiseScales,
			                 genAct, xNorms, means, invStds, noiseVecs);
			outSamples[s] = genAct.back();
		}
		return NNetworkStatus();
	}

	outSamples.resize(numSamples);
	for (unsigned int s = 0; s < numSamples; ++s)
	{
		std::vector<float> noise;
		sampleNoise(noise);

		if (isDeconv)
		{
			std::vector<float> output;
			deconvForward(generator, &noise[0], config.noiseDim, output);
			outSamples[s] = output;
		}
		else
		{
			std::vector<std::vector<float> > genAct;
			dffForward(generator, &noise[0], config.noiseDim, genAct, config.generatorOutputSigmoid, genLN());
			outSamples[s] = genAct.back();
		}
	}

	return NNetworkStatus();
}

// ============================================================
// InfoGAN: Latent code sampling
// ============================================================

void GAN::sampleLatentCodes(std::vector<float>& catCode, std::vector<float>& contCode) const
{
	const unsigned int numCat = config.infoConfig.numCategorical;
	const unsigned int numCont = config.infoConfig.numContinuous;

	glades::rng::Engine& eng = const_cast<glades::rng::Engine&>(rngEngine);

	// One-hot categorical code
	catCode.assign(numCat, 0.0f);
	if (numCat > 0u)
	{
		const unsigned int idx = static_cast<unsigned int>(glades::rng::uniform_int(eng, 0, static_cast<int>(numCat) - 1));
		catCode[idx] = 1.0f;
	}

	// Continuous codes: uniform [-1, 1]
	contCode.resize(numCont);
	for (unsigned int i = 0; i < numCont; ++i)
		contCode[i] = static_cast<float>(glades::rng::uniform_double(eng, -1.0, 1.0));
}

// ============================================================
// InfoGAN: Q-network head
// ============================================================

void GAN::initQHead(unsigned int sharedDim, QNetworkHead& head, AdamState& adamSt)
{
	const unsigned int numCat = config.infoConfig.numCategorical;
	const unsigned int numCont = config.infoConfig.numContinuous;
	const unsigned int qOut = numCat + 2u * numCont; // cat logits + (mu, logvar) per cont

	head.sharedDim = sharedDim;
	head.qOutDim = qOut;
	head.W.assign(static_cast<size_t>(qOut) * sharedDim, 0.0f);
	head.bias.assign(qOut, 0.0f);
	head.gW.assign(head.W.size(), 0.0f);
	head.gBias.assign(qOut, 0.0f);

	initGlorot(rngEngine, head.W, sharedDim, qOut);

	// Init Adam state for Q-head: single "transition"
	adamSt.step = 0ULL;
	adamSt.mW.resize(1u);
	adamSt.vW.resize(1u);
	adamSt.mBias.resize(1u);
	adamSt.vBias.resize(1u);
	adamSt.mW[0].assign(head.W.size(), 0.0f);
	adamSt.vW[0].assign(head.W.size(), 0.0f);
	adamSt.mBias[0].assign(qOut, 0.0f);
	adamSt.vBias[0].assign(qOut, 0.0f);

	head.initialized = true;
}

void GAN::initGenQHead(unsigned int inputDim, unsigned int hiddenDim,
                       QNetworkHead& head, AdamState& adamSt)
{
	const unsigned int numCat = config.infoConfig.numCategorical;
	const unsigned int numCont = config.infoConfig.numContinuous;
	const unsigned int qOut = numCat + 2u * numCont;

	head.sharedDim = inputDim;
	head.qOutDim = qOut;
	head.hiddenDim = hiddenDim;

	// Hidden layer: inputDim -> hiddenDim
	head.hiddenW.assign(static_cast<size_t>(hiddenDim) * inputDim, 0.0f);
	head.hiddenBias.assign(hiddenDim, 0.0f);
	head.gHiddenW.assign(head.hiddenW.size(), 0.0f);
	head.gHiddenBias.assign(hiddenDim, 0.0f);
	initGlorot(rngEngine, head.hiddenW, inputDim, hiddenDim);

	// Output layer: hiddenDim -> qOut
	head.W.assign(static_cast<size_t>(qOut) * hiddenDim, 0.0f);
	head.bias.assign(qOut, 0.0f);
	head.gW.assign(head.W.size(), 0.0f);
	head.gBias.assign(qOut, 0.0f);
	initGlorot(rngEngine, head.W, hiddenDim, qOut);

	// Adam state: 2 transitions (hidden + output)
	adamSt.step = 0ULL;
	adamSt.mW.resize(2u);
	adamSt.vW.resize(2u);
	adamSt.mBias.resize(2u);
	adamSt.vBias.resize(2u);
	adamSt.mW[0].assign(head.hiddenW.size(), 0.0f);
	adamSt.vW[0].assign(head.hiddenW.size(), 0.0f);
	adamSt.mBias[0].assign(hiddenDim, 0.0f);
	adamSt.vBias[0].assign(hiddenDim, 0.0f);
	adamSt.mW[1].assign(head.W.size(), 0.0f);
	adamSt.vW[1].assign(head.W.size(), 0.0f);
	adamSt.mBias[1].assign(qOut, 0.0f);
	adamSt.vBias[1].assign(qOut, 0.0f);

	head.initialized = true;
}

void GAN::qHeadForward(const QNetworkHead& head, const float* shared, unsigned int dim,
                       std::vector<float>& qOut,
                       std::vector<float>* hiddenPre,
                       std::vector<float>* hiddenPost) const
{
	if (head.hiddenDim > 0u && hiddenPre && hiddenPost)
	{
		// Two-layer: shared -> hidden (LeakyReLU) -> output
		const float alpha = 0.01f;
		for (unsigned int k = 0; k < head.hiddenDim; ++k)
		{
			float z = head.hiddenBias[k];
			const size_t rowOff = static_cast<size_t>(k) * head.sharedDim;
			for (unsigned int i = 0; i < dim && i < head.sharedDim; ++i)
				z += head.hiddenW[rowOff + i] * shared[i];
			(*hiddenPre)[k] = z;
			(*hiddenPost)[k] = (z > 0.0f) ? z : alpha * z;
		}

		qOut.assign(head.qOutDim, 0.0f);
		for (unsigned int j = 0; j < head.qOutDim; ++j)
		{
			float z = head.bias[j];
			const size_t rowOff = static_cast<size_t>(j) * head.hiddenDim;
			for (unsigned int k = 0; k < head.hiddenDim; ++k)
				z += head.W[rowOff + k] * (*hiddenPost)[k];
			qOut[j] = z;
		}
	}
	else
	{
		// Single-layer (original path for disc-side qHead)
		qOut.assign(head.qOutDim, 0.0f);
		for (unsigned int j = 0; j < head.qOutDim; ++j)
		{
			float z = head.bias[j];
			const size_t rowOff = static_cast<size_t>(j) * head.sharedDim;
			for (unsigned int i = 0; i < dim && i < head.sharedDim; ++i)
				z += head.W[rowOff + i] * shared[i];
			qOut[j] = z;
		}
	}
}

void GAN::qHeadBackward(QNetworkHead& head, const float* shared, const float* qGrad,
                        std::vector<float>& sharedGrad)
{
	sharedGrad.assign(head.sharedDim, 0.0f);
	for (unsigned int j = 0; j < head.qOutDim; ++j)
	{
		const float d = qGrad[j];
		head.gBias[j] += d;
		const size_t rowOff = static_cast<size_t>(j) * head.sharedDim;
		for (unsigned int i = 0; i < head.sharedDim; ++i)
		{
			head.gW[rowOff + i] += d * shared[i];
			sharedGrad[i] += d * head.W[rowOff + i];
		}
	}
}

void GAN::qHeadBackward(const QNetworkHead& head, const float* shared, const float* qGrad,
                        std::vector<float>& sharedGrad,
                        GradientBuffer& gradBuf)
{
	sharedGrad.assign(head.sharedDim, 0.0f);
	for (unsigned int j = 0; j < head.qOutDim; ++j)
	{
		const float d = qGrad[j];
		gradBuf.qGBias[j] += d;
		const size_t rowOff = static_cast<size_t>(j) * head.sharedDim;
		for (unsigned int i = 0; i < head.sharedDim; ++i)
		{
			gradBuf.qGW[rowOff + i] += d * shared[i];
			sharedGrad[i] += d * head.W[rowOff + i];
		}
	}
}

void GAN::qHeadUpdate(QNetworkHead& head, AdamState& adamSt, float lr)
{
	++adamSt.step;
	const float beta1 = config.adamBeta1;
	const float beta2 = config.adamBeta2;
	const float eps = config.adamEps;
	const float bc1 = 1.0f - powf(beta1, static_cast<float>(adamSt.step));
	const float bc2 = 1.0f - powf(beta2, static_cast<float>(adamSt.step));

	for (size_t i = 0; i < head.W.size(); ++i)
	{
		const float g = head.gW[i];
		adamSt.mW[0][i] = beta1 * adamSt.mW[0][i] + (1.0f - beta1) * g;
		adamSt.vW[0][i] = beta2 * adamSt.vW[0][i] + (1.0f - beta2) * g * g;
		const float mHat = adamSt.mW[0][i] / bc1;
		const float vHat = adamSt.vW[0][i] / bc2;
		head.W[i] -= lr * mHat / (sqrtf(vHat) + eps);
		head.gW[i] = 0.0f;
	}
	for (unsigned int j = 0; j < head.qOutDim; ++j)
	{
		const float g = head.gBias[j];
		adamSt.mBias[0][j] = beta1 * adamSt.mBias[0][j] + (1.0f - beta1) * g;
		adamSt.vBias[0][j] = beta2 * adamSt.vBias[0][j] + (1.0f - beta2) * g * g;
		const float mHat = adamSt.mBias[0][j] / bc1;
		const float vHat = adamSt.vBias[0][j] / bc2;
		head.bias[j] -= lr * mHat / (sqrtf(vHat) + eps);
		head.gBias[j] = 0.0f;
	}
}

void GAN::genQHeadUpdate(QNetworkHead& head, AdamState& adamSt, float lr)
{
	++adamSt.step;
	const float beta1 = config.adamBeta1;
	const float beta2 = config.adamBeta2;
	const float eps = config.adamEps;
	const float bc1 = 1.0f - powf(beta1, static_cast<float>(adamSt.step));
	const float bc2 = 1.0f - powf(beta2, static_cast<float>(adamSt.step));

	// Hidden layer (transition 0)
	if (head.hiddenDim > 0u)
	{
		for (size_t i = 0; i < head.hiddenW.size(); ++i)
		{
			const float g = head.gHiddenW[i];
			adamSt.mW[0][i] = beta1 * adamSt.mW[0][i] + (1.0f - beta1) * g;
			adamSt.vW[0][i] = beta2 * adamSt.vW[0][i] + (1.0f - beta2) * g * g;
			head.hiddenW[i] -= lr * (adamSt.mW[0][i] / bc1) / (sqrtf(adamSt.vW[0][i] / bc2) + eps);
			head.gHiddenW[i] = 0.0f;
		}
		for (unsigned int k = 0; k < head.hiddenDim; ++k)
		{
			const float g = head.gHiddenBias[k];
			adamSt.mBias[0][k] = beta1 * adamSt.mBias[0][k] + (1.0f - beta1) * g;
			adamSt.vBias[0][k] = beta2 * adamSt.vBias[0][k] + (1.0f - beta2) * g * g;
			head.hiddenBias[k] -= lr * (adamSt.mBias[0][k] / bc1) / (sqrtf(adamSt.vBias[0][k] / bc2) + eps);
			head.gHiddenBias[k] = 0.0f;
		}
	}

	// Output layer (transition 1 if hidden, 0 if no hidden)
	const unsigned int outIdx = (head.hiddenDim > 0u) ? 1u : 0u;
	for (size_t i = 0; i < head.W.size(); ++i)
	{
		const float g = head.gW[i];
		adamSt.mW[outIdx][i] = beta1 * adamSt.mW[outIdx][i] + (1.0f - beta1) * g;
		adamSt.vW[outIdx][i] = beta2 * adamSt.vW[outIdx][i] + (1.0f - beta2) * g * g;
		head.W[i] -= lr * (adamSt.mW[outIdx][i] / bc1) / (sqrtf(adamSt.vW[outIdx][i] / bc2) + eps);
		head.gW[i] = 0.0f;
	}
	for (unsigned int j = 0; j < head.qOutDim; ++j)
	{
		const float g = head.gBias[j];
		adamSt.mBias[outIdx][j] = beta1 * adamSt.mBias[outIdx][j] + (1.0f - beta1) * g;
		adamSt.vBias[outIdx][j] = beta2 * adamSt.vBias[outIdx][j] + (1.0f - beta2) * g * g;
		head.bias[j] -= lr * (adamSt.mBias[outIdx][j] / bc1) / (sqrtf(adamSt.vBias[outIdx][j] / bc2) + eps);
		head.gBias[j] = 0.0f;
	}
}

float GAN::computeInfoLoss(const std::vector<float>& qOut,
                           const std::vector<float>& catCode, const std::vector<float>& contCode,
                           std::vector<float>& qGrad) const
{
	const unsigned int numCat = config.infoConfig.numCategorical;
	const unsigned int numCont = config.infoConfig.numContinuous;

	qGrad.assign(qHead.qOutDim, 0.0f);
	float loss = 0.0f;

	// Categorical: softmax + cross-entropy
	if (numCat > 0u)
	{
		std::vector<float> catLogits(numCat);
		for (unsigned int i = 0; i < numCat; ++i)
			catLogits[i] = qOut[i];
		softmaxVec(&catLogits[0], numCat);

		loss += categoricalCrossEntropy(&catLogits[0], &catCode[0], numCat);
		categoricalCEGrad(&catLogits[0], &catCode[0], &qGrad[0], numCat);
	}

	// Continuous: Gaussian NLL (clamp logvar to prevent continuous gradient
	// from dominating categorical gradient through the shared Q-head path)
	for (unsigned int c = 0; c < numCont; ++c)
	{
		const unsigned int muIdx = numCat + 2u * c;
		const unsigned int lvIdx = numCat + 2u * c + 1u;
		const float mu = qOut[muIdx];
		const float logvar = (qOut[lvIdx] < -2.0f) ? -2.0f : qOut[lvIdx];
		loss += gaussianNLL(mu, logvar, contCode[c]);

		float dMu, dLogvar;
		gaussianNLLGrad(mu, logvar, contCode[c], dMu, dLogvar);
		qGrad[muIdx] = dMu;
		qGrad[lvIdx] = (qOut[lvIdx] < -2.0f) ? 0.0f : dLogvar;
	}

	return loss;
}

// ============================================================
// InfoGAN: Controlled generation
// ============================================================

NNetworkStatus GAN::generateInfoGAN(unsigned int numSamples,
                                    const std::vector<float>* fixedCatCode,
                                    const std::vector<float>* fixedContCode,
                                    std::vector<std::vector<float> >& outSamples) const
{
	const bool isDeconv = generator.tensorDeconv.initialized;
	if (!generator.tensorDff.initialized && !isDeconv)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "GAN::generateInfoGAN: generator not trained");

	const bool hasStyle = config.useStyle || (config.variantType == GANConfig::GAN_STYLE);
	const unsigned int numCat = config.infoConfig.numCategorical;
	const unsigned int numCont = config.infoConfig.numContinuous;
	const unsigned int totalInputDim = config.noiseDim + numCat + numCont;

	outSamples.resize(numSamples);
	for (unsigned int s = 0; s < numSamples; ++s)
	{
		std::vector<float> noise;
		sampleNoise(noise);

		std::vector<float> catCode, contCode;
		if (fixedCatCode)
			catCode = *fixedCatCode;
		else
			sampleLatentCodes(catCode, contCode);

		if (fixedContCode)
			contCode = *fixedContCode;
		else if (!fixedCatCode)
		{
			// already sampled above
		}
		else
			sampleLatentCodes(catCode, contCode); // just need contCode

		catCode.resize(numCat, 0.0f);
		contCode.resize(numCont, 0.0f);

		std::vector<float> genInput(totalInputDim);
		std::memcpy(&genInput[0], &noise[0], config.noiseDim * sizeof(float));
		if (numCat > 0u)
		{
			for (unsigned int c = 0; c < numCat; ++c)
				genInput[config.noiseDim + c] = catCode[c] * config.infoConfig.catScale;
		}
		if (numCont > 0u)
			std::memcpy(&genInput[config.noiseDim + numCat], &contCode[0], numCont * sizeof(float));

		if (isDeconv)
		{
			std::vector<float> output;
			deconvForward(generator, &genInput[0], totalInputDim, output);
			outSamples[s] = output;
		}
		else if (hasStyle && mappingNet.tensorDff.initialized)
		{
			std::vector<std::vector<float> > mapAct;
			dffForward(mappingNet, &noise[0], config.noiseDim, mapAct);
			const std::vector<float>& w = mapAct.back();

			std::vector<std::vector<float> > genAct, xN;
			std::vector<float> gM, gI;
			std::vector<std::vector<float> > gNV;
			dffForwardStyled(generator, &genInput[0], totalInputDim, w,
			                 styleAffines, noiseScales,
			                 genAct, xN, gM, gI, gNV);
			outSamples[s] = genAct.back();
		}
		else
		{
			std::vector<std::vector<float> > genAct;
			dffForward(generator, &genInput[0], totalInputDim, genAct, config.generatorOutputSigmoid, genLN());
			outSamples[s] = genAct.back();
		}
	}

	return NNetworkStatus();
}

// ============================================================
// StyleGAN: Style affine initialization
// ============================================================

void GAN::initStyleAffines(unsigned int wDim, const NNetwork& gen,
                           std::vector<StyleAffine>& affines, AdamState& adamSt,
                           std::vector<float>& scales, std::vector<float>& gScales)
{
	const unsigned int numT = static_cast<unsigned int>(gen.tensorDff.T.size());
	if (numT < 2u)
		return;

	// Style affines for hidden layers (all transitions except the last one)
	const unsigned int numStyleLayers = numT - 1u;
	affines.resize(numStyleLayers);
	scales.assign(numStyleLayers, config.styleConfig.noiseScaleInit);
	gScales.assign(numStyleLayers, 0.0f);

	for (unsigned int t = 0; t < numStyleLayers; ++t)
	{
		const unsigned int layerSize = gen.tensorDff.T[t].out;
		const unsigned int affineOut = 2u * layerSize; // scale + bias

		StyleAffine& sa = affines[t];
		sa.wDim = wDim;
		sa.outDim = affineOut;
		sa.W.assign(static_cast<size_t>(affineOut) * wDim, 0.0f);
		sa.bias.assign(affineOut, 0.0f);
		sa.gW.assign(sa.W.size(), 0.0f);
		sa.gBias.assign(affineOut, 0.0f);

		initGlorot(rngEngine, sa.W, wDim, affineOut);

		// Initialize bias: scale part to 1.0, bias part to 0.0
		for (unsigned int j = 0; j < layerSize; ++j)
			sa.bias[j] = 1.0f;
	}

	// Adam state for style affines (one set of m/v per affine)
	adamSt.step = 0ULL;
	adamSt.mW.resize(numStyleLayers);
	adamSt.vW.resize(numStyleLayers);
	adamSt.mBias.resize(numStyleLayers);
	adamSt.vBias.resize(numStyleLayers);
	for (unsigned int t = 0; t < numStyleLayers; ++t)
	{
		adamSt.mW[t].assign(affines[t].W.size(), 0.0f);
		adamSt.vW[t].assign(affines[t].W.size(), 0.0f);
		adamSt.mBias[t].assign(affines[t].bias.size(), 0.0f);
		adamSt.vBias[t].assign(affines[t].bias.size(), 0.0f);
	}
}

// ============================================================
// StyleGAN: Styled forward pass
// ============================================================

void GAN::dffForwardStyled(const NNetwork& net, const float* input, unsigned int inputSize,
                           const std::vector<float>& w,
                           const std::vector<StyleAffine>& affines,
                           const std::vector<float>& scales,
                           std::vector<std::vector<float> >& activations,
                           std::vector<std::vector<float> >& xNorms,
                           std::vector<float>& means, std::vector<float>& invStds,
                           std::vector<std::vector<float> >& noiseVecs) const
{
	const NNetwork::TensorDFFState& dff = net.tensorDff;
	const NNInfo* skel = net.skeleton;
	const unsigned int numLayers = static_cast<unsigned int>(dff.sizes.size());
	const unsigned int numT = static_cast<unsigned int>(dff.T.size());
	const unsigned int numStyleLayers = (numT >= 1u) ? numT - 1u : 0u;

	activations.resize(numLayers);
	xNorms.resize(numStyleLayers);
	means.resize(numStyleLayers);
	invStds.resize(numStyleLayers);
	noiseVecs.resize(numStyleLayers);

	activations[0].assign(input, input + inputSize);

	glades::rng::Engine& eng = const_cast<glades::rng::Engine&>(rngEngine);

	std::vector<float> styleOut;

	for (unsigned int t = 0; t < numT; ++t)
	{
		const NNetwork::TensorDFFState::Transition& tr = dff.T[t];
		const int actFx = skel->getActivationType(t);
		const float actParam = skel->getActivationParam(t);
		const bool isLastLayer = (t == numT - 1u);

		activations[t + 1u].assign(tr.out, 0.0f);

		// Linear: z = W * a[t] + bias (SIMD GEMV)
		transformer_kernels::gemv_rowmajor_bias_block4_unroll8_into(
			activations[t].data(), tr.in,
			tr.W.data(), tr.out,
			tr.bias.data(), tr.out,
			&activations[t + 1u][0]);

		if (isLastLayer)
		{
			// Last layer: sigmoid if configured (for [0,1] data like images)
			for (unsigned int j = 0; j < tr.out; ++j)
			{
				if (config.generatorOutputSigmoid)
					activations[t + 1u][j] = sigmoid(activations[t + 1u][j]);
				else
					activations[t + 1u][j] = GMath::squash(activations[t + 1u][j], actFx, actParam);
			}
		}
		else
		{
			// Hidden layer: activation -> instance norm -> AdaIN -> noise
			for (unsigned int j = 0; j < tr.out; ++j)
				activations[t + 1u][j] = GMath::squash(activations[t + 1u][j], actFx, actParam);

			// Instance normalization
			xNorms[t].resize(tr.out);
			instanceNorm(&activations[t + 1u][0], tr.out, 1e-5f,
			             &xNorms[t][0], means[t], invStds[t]);

			// Style affine: [y_s, y_b] = W_style * w + bias_style
			const StyleAffine& sa = affines[t];
			const unsigned int layerSize = tr.out;
			styleOut.assign(sa.outDim, 0.0f);
			const unsigned int safeDim = std::min(sa.wDim, static_cast<unsigned int>(w.size()));
			transformer_kernels::gemv_rowmajor_bias_block4_unroll8_into(
				w.data(), safeDim,
				sa.W.data(), sa.outDim,
				sa.bias.data(), sa.outDim,
				styleOut.data());

			// AdaIN: out = y_s * x_norm + y_b
			for (unsigned int j = 0; j < layerSize; ++j)
			{
				const float ys = styleOut[j];
				const float yb = styleOut[layerSize + j];
				activations[t + 1u][j] = ys * xNorms[t][j] + yb;
			}

			// Noise injection
			noiseVecs[t].resize(layerSize);
			for (unsigned int j = 0; j < layerSize; ++j)
				noiseVecs[t][j] = glades::rng::standard_normal(eng);
			for (unsigned int j = 0; j < layerSize; ++j)
				activations[t + 1u][j] += scales[t] * noiseVecs[t][j];
		}
	}
}

// ============================================================
// StyleGAN: Styled backward pass
// ============================================================

void GAN::dffBackwardStyled(NNetwork& net, const std::vector<std::vector<float> >& activations,
                            const float* outputGrad, unsigned int outputSize,
                            const std::vector<float>& w,
                            const std::vector<std::vector<float> >& xNorms,
                            const std::vector<float>& means, const std::vector<float>& invStds,
                            const std::vector<std::vector<float> >& noiseVecs,
                            std::vector<StyleAffine>& affines,
                            std::vector<float>& gScales,
                            std::vector<float>& dW)
{
	NNetwork::TensorDFFState& dff = net.tensorDff;
	const NNInfo* skel = net.skeleton;
	const unsigned int numLayers = static_cast<unsigned int>(dff.sizes.size());
	const unsigned int numT = static_cast<unsigned int>(dff.T.size());
	const unsigned int numStyleLayers = (numT >= 1u) ? numT - 1u : 0u;

	if (numLayers < 2u)
		return;

	const unsigned int wDim = (affines.empty()) ? 0u : affines[0].wDim;
	dW.assign(wDim, 0.0f);

	// Compute deltas - reuse class scratch buffer
	std::vector<std::vector<float> >& delta = scratchDelta;
	delta.resize(numLayers);
	for (unsigned int li = 0; li < numLayers; ++li)
		delta[li].assign(dff.sizes[li], 0.0f);

	// Output delta
	const unsigned int lastLayer = numLayers - 1u;
	const unsigned int lastTransition = numT - 1u;
	for (unsigned int k = 0; k < outputSize; ++k)
	{
		const float a = activations[lastLayer][k];
		if (config.generatorOutputSigmoid)
			delta[lastLayer][k] = outputGrad[k] * sigmoidDeriv(a);
		else
		{
			const int actFx = skel->getActivationType(lastTransition);
			const float actParam = skel->getActivationParam(lastTransition);
			delta[lastLayer][k] = outputGrad[k] * GMath::activationErrDer(a, actFx, actParam);
		}
	}

	// Backward through layers
	std::vector<float> dPreStyle;
	std::vector<float> styleOut;
	std::vector<float> dYS, dYB, dXNorm;
	std::vector<float> dActRaw;

	for (int li = static_cast<int>(lastLayer) - 1; li >= 1; --li)
	{
		const unsigned int l = static_cast<unsigned int>(li);
		const NNetwork::TensorDFFState::Transition& nextTr = dff.T[l];
		const unsigned int layerSize = dff.sizes[l];

		// Propagate delta from layer l+1 to layer l (row-wise axpy)
		dPreStyle.assign(layerSize, 0.0f);
		for (unsigned int j = 0; j < nextTr.out; ++j)
			transformer_kernels::axpy_f32(
				dPreStyle.data(),
				&nextTr.W[static_cast<size_t>(j) * nextTr.in],
				delta[l + 1u][j], nextTr.in);

		// Styled layer backward
		if (l >= 1u && l - 1u < numStyleLayers)
		{
			const unsigned int styleIdx = l - 1u;
			const StyleAffine& sa = affines[styleIdx];

			// Noise scale gradient
			float dNoiseScale = 0.0f;
			for (unsigned int j = 0; j < layerSize; ++j)
				dNoiseScale += dPreStyle[j] * noiseVecs[styleIdx][j];
			gScales[styleIdx] += dNoiseScale;

			// Reconstruct style outputs (SIMD GEMV)
			styleOut.assign(sa.outDim, 0.0f);
			const unsigned int safeDim = std::min(sa.wDim, static_cast<unsigned int>(w.size()));
			transformer_kernels::gemv_rowmajor_bias_block4_unroll8_into(
				w.data(), safeDim,
				sa.W.data(), sa.outDim,
				sa.bias.data(), sa.outDim,
				styleOut.data());

			// AdaIN backward
			dYS.assign(layerSize, 0.0f);
			dYB.assign(layerSize, 0.0f);
			dXNorm.assign(layerSize, 0.0f);
			for (unsigned int j = 0; j < layerSize; ++j)
			{
				dYS[j] = dPreStyle[j] * xNorms[styleIdx][j];
				dYB[j] = dPreStyle[j];
				dXNorm[j] = dPreStyle[j] * styleOut[j];
			}

			// Style affine gradients (SIMD axpy)
			StyleAffine& saMut = affines[styleIdx];
			for (unsigned int j = 0; j < layerSize; ++j)
			{
				saMut.gBias[j] += dYS[j];
				saMut.gBias[layerSize + j] += dYB[j];
				transformer_kernels::axpy_f32(
					&saMut.gW[static_cast<size_t>(j) * sa.wDim],
					w.data(), dYS[j], safeDim);
				transformer_kernels::axpy_f32(
					&saMut.gW[static_cast<size_t>(layerSize + j) * sa.wDim],
					w.data(), dYB[j], safeDim);
			}

			// Accumulate dW (row-wise axpy)
			for (unsigned int j = 0; j < layerSize; ++j)
			{
				transformer_kernels::axpy_f32(
					dW.data(),
					&sa.W[static_cast<size_t>(j) * sa.wDim],
					dYS[j], safeDim);
				transformer_kernels::axpy_f32(
					dW.data(),
					&sa.W[static_cast<size_t>(layerSize + j) * sa.wDim],
					dYB[j], safeDim);
			}

			// Instance norm backward
			dActRaw.assign(layerSize, 0.0f);
			instanceNormBackward(&dXNorm[0], &xNorms[styleIdx][0],
			                     invStds[styleIdx], layerSize, &dActRaw[0]);

			// Activation derivative
			const int actFx = skel->getActivationType(l - 1u);
			const float actParam = skel->getActivationParam(l - 1u);
			for (unsigned int i = 0; i < layerSize; ++i)
			{
				const float aRaw = xNorms[styleIdx][i] / invStds[styleIdx] + means[styleIdx];
				const float dA_dZ = GMath::activationErrDer(aRaw, actFx, actParam);
				delta[l][i] = dActRaw[i] * dA_dZ;
			}
		}
		else
		{
			const int actFx = skel->getActivationType(l - 1u);
			const float actParam = skel->getActivationParam(l - 1u);
			for (unsigned int i = 0; i < layerSize; ++i)
			{
				const float dA_dZ = GMath::activationErrDer(activations[l][i], actFx, actParam);
				delta[l][i] = dPreStyle[i] * dA_dZ;
			}
		}
	}

	// Accumulate weight gradients (SIMD axpy)
	for (unsigned int t = 0; t < numT; ++t)
	{
		NNetwork::TensorDFFState::Transition& tr = dff.T[t];
		for (unsigned int j = 0; j < tr.out; ++j)
		{
			tr.gBias[j] += delta[t + 1u][j];
			transformer_kernels::axpy_f32(
				&tr.gW[static_cast<size_t>(j) * tr.in],
				activations[t].data(),
				delta[t + 1u][j], tr.in);
		}
	}
}

// ============================================================
// StyleGAN: Styled backward pass (GradientBuffer overload - thread-safe)
// ============================================================

void GAN::dffBackwardStyled(const NNetwork& net, const std::vector<std::vector<float> >& activations,
                            const float* outputGrad, unsigned int outputSize,
                            const std::vector<float>& w,
                            const std::vector<std::vector<float> >& xNorms,
                            const std::vector<float>& means, const std::vector<float>& invStds,
                            const std::vector<std::vector<float> >& noiseVecs,
                            const std::vector<StyleAffine>& affines,
                            std::vector<float>& gScalesLocal,
                            std::vector<float>& dW,
                            GradientBuffer& gradBuf,
                            std::vector<std::vector<float> >& styleGW,
                            std::vector<std::vector<float> >& styleGBias,
                            std::vector<std::vector<float> >& scratchDeltaLocal)
{
	const NNetwork::TensorDFFState& dff = net.tensorDff;
	const NNInfo* skel = net.skeleton;
	const unsigned int numLayers = static_cast<unsigned int>(dff.sizes.size());
	const unsigned int numT = static_cast<unsigned int>(dff.T.size());
	const unsigned int numStyleLayers = (numT >= 1u) ? numT - 1u : 0u;

	if (numLayers < 2u)
		return;

	const unsigned int wDim = (affines.empty()) ? 0u : affines[0].wDim;
	dW.assign(wDim, 0.0f);

	// Compute deltas - use caller-provided scratch buffer
	std::vector<std::vector<float> >& delta = scratchDeltaLocal;
	delta.resize(numLayers);
	for (unsigned int li = 0; li < numLayers; ++li)
		delta[li].assign(dff.sizes[li], 0.0f);

	// Output delta
	const unsigned int lastLayer = numLayers - 1u;
	const unsigned int lastTransition = numT - 1u;
	for (unsigned int k = 0; k < outputSize; ++k)
	{
		const float a = activations[lastLayer][k];
		if (config.generatorOutputSigmoid)
			delta[lastLayer][k] = outputGrad[k] * sigmoidDeriv(a);
		else
		{
			const int actFx = skel->getActivationType(lastTransition);
			const float actParam = skel->getActivationParam(lastTransition);
			delta[lastLayer][k] = outputGrad[k] * GMath::activationErrDer(a, actFx, actParam);
		}
	}

	// Backward through layers
	std::vector<float> dPreStyle;
	std::vector<float> styleOut;
	std::vector<float> dYS, dYB, dXNorm;
	std::vector<float> dActRaw;

	for (int li = static_cast<int>(lastLayer) - 1; li >= 1; --li)
	{
		const unsigned int l = static_cast<unsigned int>(li);
		const NNetwork::TensorDFFState::Transition& nextTr = dff.T[l];
		const unsigned int layerSize = dff.sizes[l];

		// Propagate delta from layer l+1 to layer l (row-wise axpy)
		dPreStyle.assign(layerSize, 0.0f);
		for (unsigned int j = 0; j < nextTr.out; ++j)
			transformer_kernels::axpy_f32(
				dPreStyle.data(),
				&nextTr.W[static_cast<size_t>(j) * nextTr.in],
				delta[l + 1u][j], nextTr.in);

		// Styled layer backward
		if (l >= 1u && l - 1u < numStyleLayers)
		{
			const unsigned int styleIdx = l - 1u;
			const StyleAffine& sa = affines[styleIdx];

			// Noise scale gradient into local buffer
			float dNoiseScale = 0.0f;
			for (unsigned int j = 0; j < layerSize; ++j)
				dNoiseScale += dPreStyle[j] * noiseVecs[styleIdx][j];
			gScalesLocal[styleIdx] += dNoiseScale;

			// Reconstruct style outputs (SIMD GEMV)
			styleOut.assign(sa.outDim, 0.0f);
			const unsigned int safeDim = std::min(sa.wDim, static_cast<unsigned int>(w.size()));
			transformer_kernels::gemv_rowmajor_bias_block4_unroll8_into(
				w.data(), safeDim,
				sa.W.data(), sa.outDim,
				sa.bias.data(), sa.outDim,
				styleOut.data());

			// AdaIN backward
			dYS.assign(layerSize, 0.0f);
			dYB.assign(layerSize, 0.0f);
			dXNorm.assign(layerSize, 0.0f);
			for (unsigned int j = 0; j < layerSize; ++j)
			{
				dYS[j] = dPreStyle[j] * xNorms[styleIdx][j];
				dYB[j] = dPreStyle[j];
				dXNorm[j] = dPreStyle[j] * styleOut[j];
			}

			// Style affine gradients into local style buffers (SIMD axpy)
			for (unsigned int j = 0; j < layerSize; ++j)
			{
				styleGBias[styleIdx][j] += dYS[j];
				styleGBias[styleIdx][layerSize + j] += dYB[j];
				transformer_kernels::axpy_f32(
					&styleGW[styleIdx][static_cast<size_t>(j) * sa.wDim],
					w.data(), dYS[j], safeDim);
				transformer_kernels::axpy_f32(
					&styleGW[styleIdx][static_cast<size_t>(layerSize + j) * sa.wDim],
					w.data(), dYB[j], safeDim);
			}

			// Accumulate dW (row-wise axpy)
			for (unsigned int j = 0; j < layerSize; ++j)
			{
				transformer_kernels::axpy_f32(
					dW.data(),
					&sa.W[static_cast<size_t>(j) * sa.wDim],
					dYS[j], safeDim);
				transformer_kernels::axpy_f32(
					dW.data(),
					&sa.W[static_cast<size_t>(layerSize + j) * sa.wDim],
					dYB[j], safeDim);
			}

			// Instance norm backward
			dActRaw.assign(layerSize, 0.0f);
			instanceNormBackward(&dXNorm[0], &xNorms[styleIdx][0],
			                     invStds[styleIdx], layerSize, &dActRaw[0]);

			// Activation derivative
			const int actFx = skel->getActivationType(l - 1u);
			const float actParam = skel->getActivationParam(l - 1u);
			for (unsigned int i = 0; i < layerSize; ++i)
			{
				const float aRaw = xNorms[styleIdx][i] / invStds[styleIdx] + means[styleIdx];
				const float dA_dZ = GMath::activationErrDer(aRaw, actFx, actParam);
				delta[l][i] = dActRaw[i] * dA_dZ;
			}
		}
		else
		{
			const int actFx = skel->getActivationType(l - 1u);
			const float actParam = skel->getActivationParam(l - 1u);
			for (unsigned int i = 0; i < layerSize; ++i)
			{
				const float dA_dZ = GMath::activationErrDer(activations[l][i], actFx, actParam);
				delta[l][i] = dPreStyle[i] * dA_dZ;
			}
		}
	}

	// Accumulate weight gradients into GradientBuffer (SIMD axpy)
	for (unsigned int t = 0; t < numT; ++t)
	{
		const NNetwork::TensorDFFState::Transition& tr = dff.T[t];
		for (unsigned int j = 0; j < tr.out; ++j)
		{
			gradBuf.dffGBias[t][j] += delta[t + 1u][j];
			transformer_kernels::axpy_f32(
				&gradBuf.dffGW[t][static_cast<size_t>(j) * tr.in],
				activations[t].data(),
				delta[t + 1u][j], tr.in);
		}
	}
}

// ============================================================
// StyleGAN: Style affine Adam update
// ============================================================

void GAN::styleAffineUpdate(float lr,
                            std::vector<StyleAffine>& affines, AdamState& adamSt,
                            std::vector<float>& scales, std::vector<float>& gScales)
{
	++adamSt.step;
	const float beta1 = config.adamBeta1;
	const float beta2 = config.adamBeta2;
	const float eps = config.adamEps;
	const float bc1 = 1.0f - powf(beta1, static_cast<float>(adamSt.step));
	const float bc2 = 1.0f - powf(beta2, static_cast<float>(adamSt.step));

	for (unsigned int t = 0; t < affines.size(); ++t)
	{
		StyleAffine& sa = affines[t];
		for (size_t i = 0; i < sa.W.size(); ++i)
		{
			const float g = sa.gW[i];
			adamSt.mW[t][i] = beta1 * adamSt.mW[t][i] + (1.0f - beta1) * g;
			adamSt.vW[t][i] = beta2 * adamSt.vW[t][i] + (1.0f - beta2) * g * g;
			const float mHat = adamSt.mW[t][i] / bc1;
			const float vHat = adamSt.vW[t][i] / bc2;
			sa.W[i] -= lr * mHat / (sqrtf(vHat) + eps);
			sa.gW[i] = 0.0f;
		}
		for (unsigned int j = 0; j < sa.outDim; ++j)
		{
			const float g = sa.gBias[j];
			adamSt.mBias[t][j] = beta1 * adamSt.mBias[t][j] + (1.0f - beta1) * g;
			adamSt.vBias[t][j] = beta2 * adamSt.vBias[t][j] + (1.0f - beta2) * g * g;
			const float mHat = adamSt.mBias[t][j] / bc1;
			const float vHat = adamSt.vBias[t][j] / bc2;
			sa.bias[j] -= lr * mHat / (sqrtf(vHat) + eps);
			sa.gBias[j] = 0.0f;
		}

		// Update noise scales with simple SGD
		scales[t] -= lr * gScales[t];
		gScales[t] = 0.0f;
	}
}

// ============================================================
// Unified Dual-Domain Training (cycle, cycle+info, cycle+style, cycle+all)
// ============================================================

NNetworkStatus GAN::trainDualDomain(const DataInput* domainA, const DataInput* domainB, IGANCallbacks* cb)
{
	const bool hasInfo = config.useInfo || (config.variantType == GANConfig::GAN_INFO);
	const bool hasStyle = config.useStyle || (config.variantType == GANConfig::GAN_STYLE);

	const unsigned int sizeA = domainA->getTrainSize();
	const unsigned int sizeB = domainB->getTrainSize();
	if (sizeA == 0u || sizeB == 0u)
		return NNetworkStatus(NNetworkStatus::EMPTY_DATA, "GAN::train: no training data");

	const float* firstRowA = NULL;
	unsigned int dimA = 0u;
	domainA->getTrainRowView(0, firstRowA, dimA);
	const float* firstRowB = NULL;
	unsigned int dimB = 0u;
	domainB->getTrainRowView(0, firstRowB, dimB);

	if (dimA == 0u || dimB == 0u)
		return NNetworkStatus(NNetworkStatus::EMPTY_DATA, "GAN::train: zero feature count");

	const unsigned int numCat = hasInfo ? config.infoConfig.numCategorical : 0u;
	const unsigned int numCont = hasInfo ? config.infoConfig.numContinuous : 0u;
	const unsigned int genABInputDim = dimA + numCat + numCont;
	const unsigned int genBAInputDim = dimB + numCat + numCont;
	const float infoLambda = config.infoConfig.infoLambda;

	// Init 4 networks: genAB (A->B), genBA (B->A), discA (real A), discB (real B)
	if (config.archType == GANConfig::GAN_DFF)
	{
		if (!initDFFTensors(generator, genABInputDim, dimB))
			return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init genAB");
		if (!initDFFTensors(generatorBA, genBAInputDim, dimA))
			return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init genBA");
	}
	else
	{
		if (!initDeconvTensors(generator, genABInputDim, config.generatorDeconv))
			return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init genAB deconv");
		if (!initDeconvTensors(generatorBA, genBAInputDim, config.generatorDeconv))
			return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init genBA deconv");
	}
	if (config.archType == GANConfig::GAN_DFF)
	{
		if (!initDFFTensors(discriminator, dimA, 1u))
			return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init discA");
		if (!initDFFTensors(discriminatorB, dimB, 1u))
			return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init discB");
	}
	else
	{
		if (!initCNNTensors(discriminator, config.discriminatorCNN, 1u))
			return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init discA CNN");
		if (!initCNNTensors(discriminatorB, config.discriminatorCNN, 1u))
			return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init discB CNN");
	}

	initAdamState(genAdamState, generator);
	initAdamState(genBAAdamState, generatorBA);
	initAdamState(discAdamState, discriminator);
	initAdamState(discBAdamState, discriminatorB);
	if (config.generatorLayerNorm && !hasStyle)
	{
		initLayerNorm(genLNParams, generator);
		initLayerNorm(genBALNParams, generatorBA);
	}

	// Style init
	unsigned int wDim = 0u;
	if (hasStyle)
	{
		wDim = config.styleConfig.mappingWidth;
		if (!initDFFTensors(mappingNet, config.noiseDim, wDim))
			return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init mapping net AB");
		if (!initDFFTensors(mappingNetBA, config.noiseDim, wDim))
			return NNetworkStatus(NNetworkStatus::BUILD_FAILED, "GAN::train: failed to init mapping net BA");
		initAdamState(mappingAdamState, mappingNet);
		initAdamState(mappingBAAdamState, mappingNetBA);
		initStyleAffines(wDim, generator, styleAffines, styleAffineAdamState, noiseScales, gNoiseScales);
		initStyleAffines(wDim, generatorBA, styleAffinesBA, styleAffineBAAdamState, noiseScalesBA, gNoiseScalesBA);
	}

	// Info init (only supported with DFF discriminators for now)
	unsigned int penultDimA = 0u;
	unsigned int penultDimB = 0u;
	if (hasInfo)
	{
		if (config.archType == GANConfig::GAN_DFF)
		{
			const unsigned int numDiscLayersA = static_cast<unsigned int>(discriminator.tensorDff.sizes.size());
			penultDimA = (numDiscLayersA >= 2u) ? discriminator.tensorDff.sizes[numDiscLayersA - 2u] : dimA;
			const unsigned int numDiscLayersB = static_cast<unsigned int>(discriminatorB.tensorDff.sizes.size());
			penultDimB = (numDiscLayersB >= 2u) ? discriminatorB.tensorDff.sizes[numDiscLayersB - 2u] : dimB;
		}
		else
		{
			// CNN: penultimate FC layer output size
			const unsigned int numCnnFCA = static_cast<unsigned int>(discriminator.tensorCnn.fcLayers.size());
			penultDimA = (numCnnFCA >= 2u) ? discriminator.tensorCnn.fcLayers[numCnnFCA - 2u].out : discriminator.tensorCnn.flattenedSize;
			const unsigned int numCnnFCB = static_cast<unsigned int>(discriminatorB.tensorCnn.fcLayers.size());
			penultDimB = (numCnnFCB >= 2u) ? discriminatorB.tensorCnn.fcLayers[numCnnFCB - 2u].out : discriminatorB.tensorCnn.flattenedSize;
		}
		initQHead(penultDimA, qHead, qAdamState);
		initQHead(penultDimB, qHeadB, qBAdamState);
	}

	const float cycleLambda = config.cycleConfig.cycleLambda;
	const float identityLambda = config.cycleConfig.identityLambda;
	const unsigned int numSamples = (sizeA < sizeB) ? sizeA : sizeB;
	const int batchSize = (config.batchSize > 0) ? config.batchSize : static_cast<int>(numSamples);
	const unsigned int numBatches = (numSamples + static_cast<unsigned int>(batchSize) - 1u) / static_cast<unsigned int>(batchSize);
	const bool genIsDeconv = (config.archType == GANConfig::GAN_CNN) && !config.generatorDeconv.layers.empty();

	// --- Parallel training setup ---
	const unsigned int nThreads = ThreadPool::instance().numThreads();
	std::vector<GANThreadCtx> threadCtx(nThreads);
	for (unsigned int tid = 0; tid < nThreads; ++tid)
	{
		// genAB input/grads
		threadCtx[tid].genInput.resize(genABInputDim, 0.0f);
		if (genIsDeconv)
			threadCtx[tid].genGrads.initFromDeconv(generator);
		else
			threadCtx[tid].genGrads.initFromDFF(generator);
		// genBA input/grads
		threadCtx[tid].genBAInput.resize(genBAInputDim, 0.0f);
		if (genIsDeconv)
			threadCtx[tid].genBAGrads.initFromDeconv(generatorBA);
		else
			threadCtx[tid].genBAGrads.initFromDFF(generatorBA);
		// discA grads
		if (config.archType == GANConfig::GAN_DFF)
			threadCtx[tid].discGrads.initFromDFF(discriminator);
		else
			threadCtx[tid].discGrads.initFromCNN(discriminator);
		// discB grads
		if (config.archType == GANConfig::GAN_DFF)
			threadCtx[tid].discBGrads.initFromDFF(discriminatorB);
		else
			threadCtx[tid].discBGrads.initFromCNN(discriminatorB);
		// Q-head grads
		if (hasInfo)
		{
			threadCtx[tid].qGrads.initFromQHead(qHead.sharedDim, qHead.qOutDim);
			threadCtx[tid].qBGrads.initFromQHead(qHeadB.sharedDim, qHeadB.qOutDim);
		}
		// Style AB
		if (hasStyle)
		{
			threadCtx[tid].mappingGrads.initFromDFF(mappingNet);
			threadCtx[tid].styleGW.resize(styleAffines.size());
			threadCtx[tid].styleGBias.resize(styleAffines.size());
			for (size_t sa = 0; sa < styleAffines.size(); ++sa)
			{
				threadCtx[tid].styleGW[sa].assign(styleAffines[sa].gW.size(), 0.0f);
				threadCtx[tid].styleGBias[sa].assign(styleAffines[sa].gBias.size(), 0.0f);
			}
			threadCtx[tid].gNoiseScalesLocal.assign(noiseScales.size(), 0.0f);
			// Style BA
			threadCtx[tid].mappingBAGrads.initFromDFF(mappingNetBA);
			threadCtx[tid].styleBAGW.resize(styleAffinesBA.size());
			threadCtx[tid].styleBAGBias.resize(styleAffinesBA.size());
			for (size_t sa = 0; sa < styleAffinesBA.size(); ++sa)
			{
				threadCtx[tid].styleBAGW[sa].assign(styleAffinesBA[sa].gW.size(), 0.0f);
				threadCtx[tid].styleBAGBias[sa].assign(styleAffinesBA[sa].gBias.size(), 0.0f);
			}
			threadCtx[tid].gNoiseScalesBALocal.assign(noiseScalesBA.size(), 0.0f);
		}
		// Discard buffers
		if (config.archType == GANConfig::GAN_DFF)
		{
			threadCtx[tid].discardDiscGrads.initFromDFF(discriminator);
			threadCtx[tid].discardDiscBGrads.initFromDFF(discriminatorB);
		}
		else
		{
			threadCtx[tid].discardDiscGrads.initFromCNN(discriminator);
			threadCtx[tid].discardDiscBGrads.initFromCNN(discriminatorB);
		}
		// Discard buffer for cycle input-grad-only backward through G_AB
		if (genIsDeconv)
			threadCtx[tid].discardGenGrads.initFromDeconv(generator);
		else
			threadCtx[tid].discardGenGrads.initFromDFF(generator);
		// LN AB gradient buffers
		if (config.generatorLayerNorm && !hasStyle && genLNParams.initialized)
		{
			threadCtx[tid].lnGGamma.resize(genLNParams.gGamma.size());
			threadCtx[tid].lnGBeta.resize(genLNParams.gBeta.size());
			for (size_t t = 0; t < genLNParams.gGamma.size(); ++t)
			{
				threadCtx[tid].lnGGamma[t].assign(genLNParams.gGamma[t].size(), 0.0f);
				threadCtx[tid].lnGBeta[t].assign(genLNParams.gBeta[t].size(), 0.0f);
			}
		}
		// LN BA gradient buffers
		if (config.generatorLayerNorm && !hasStyle && genBALNParams.initialized)
		{
			threadCtx[tid].lnBAGGamma.resize(genBALNParams.gGamma.size());
			threadCtx[tid].lnBAGBeta.resize(genBALNParams.gBeta.size());
			for (size_t t = 0; t < genBALNParams.gGamma.size(); ++t)
			{
				threadCtx[tid].lnBAGGamma[t].assign(genBALNParams.gGamma[t].size(), 0.0f);
				threadCtx[tid].lnBAGBeta[t].assign(genBALNParams.gBeta[t].size(), 0.0f);
			}
		}
	}

	// Pre-allocation buffers for batch data
	std::vector<const float*> preRealPtrsA, preRealPtrsB;
	std::vector<unsigned int> preRealSizesA, preRealSizesB;
	std::vector<std::vector<float> > preCatCode, preContCode;
	std::vector<std::vector<float> > preStyleNoise;
	std::vector<float> preGPEpsilon;
	std::vector<unsigned int> beginToTid;
	// Generator loop pre-alloc
	std::vector<std::vector<float> > preCatCodeAB, preContCodeAB, preCatCodeBA, preContCodeBA;
	std::vector<std::vector<float> > preStyleNoiseAB, preStyleNoiseBA;
	std::vector<std::vector<float> > preStyleNoiseRecA, preStyleNoiseRecB;
	std::vector<std::vector<float> > preStyleNoiseIdentB, preStyleNoiseIdentA;

	// Per-thread LN forward scratch
	const bool hasLN = config.generatorLayerNorm && !hasStyle && genLNParams.initialized;
	const bool hasBALN = config.generatorLayerNorm && !hasStyle && genBALNParams.initialized;
	std::vector<LayerNormParams> genLNScratch, genBALNScratch;
	if (hasLN)
	{
		genLNScratch.resize(nThreads);
		for (unsigned int tid = 0; tid < nThreads; ++tid)
		{
			genLNScratch[tid].gamma = genLNParams.gamma;
			genLNScratch[tid].beta = genLNParams.beta;
			genLNScratch[tid].zNorm.resize(genLNParams.zNorm.size());
			genLNScratch[tid].invStd.resize(genLNParams.invStd.size(), 0.0f);
			for (size_t t = 0; t < genLNParams.zNorm.size(); ++t)
				genLNScratch[tid].zNorm[t].assign(genLNParams.zNorm[t].size(), 0.0f);
			genLNScratch[tid].gGamma.resize(genLNParams.gGamma.size());
			genLNScratch[tid].gBeta.resize(genLNParams.gBeta.size());
			for (size_t t = 0; t < genLNParams.gGamma.size(); ++t)
			{
				genLNScratch[tid].gGamma[t].assign(genLNParams.gGamma[t].size(), 0.0f);
				genLNScratch[tid].gBeta[t].assign(genLNParams.gBeta[t].size(), 0.0f);
			}
			genLNScratch[tid].initialized = true;
		}
	}
	if (hasBALN)
	{
		genBALNScratch.resize(nThreads);
		for (unsigned int tid = 0; tid < nThreads; ++tid)
		{
			genBALNScratch[tid].gamma = genBALNParams.gamma;
			genBALNScratch[tid].beta = genBALNParams.beta;
			genBALNScratch[tid].zNorm.resize(genBALNParams.zNorm.size());
			genBALNScratch[tid].invStd.resize(genBALNParams.invStd.size(), 0.0f);
			for (size_t t = 0; t < genBALNParams.zNorm.size(); ++t)
				genBALNScratch[tid].zNorm[t].assign(genBALNParams.zNorm[t].size(), 0.0f);
			genBALNScratch[tid].gGamma.resize(genBALNParams.gGamma.size());
			genBALNScratch[tid].gBeta.resize(genBALNParams.gBeta.size());
			for (size_t t = 0; t < genBALNParams.gGamma.size(); ++t)
			{
				genBALNScratch[tid].gGamma[t].assign(genBALNParams.gGamma[t].size(), 0.0f);
				genBALNScratch[tid].gBeta[t].assign(genBALNParams.gBeta[t].size(), 0.0f);
			}
			genBALNScratch[tid].initialized = true;
		}
	}

	for (int epoch = 0; epoch < config.epochs; ++epoch)
	{
		float epochDLossA = 0.0f;
		float epochDLossB = 0.0f;
		float epochGLossAB = 0.0f;
		float epochGLossBA = 0.0f;
		float epochCycleLoss = 0.0f;
		float epochIdentityLoss = 0.0f;
		float epochInfoLoss = 0.0f;
		float epochWasserstein = 0.0f;
		unsigned int epochCatCorrect = 0u;
		unsigned int epochCatTotal = 0u;
		unsigned int dataIdx = 0u;

		for (unsigned int batch = 0; batch < numBatches; ++batch)
		{
			const unsigned int curBatchSize =
			    (static_cast<int>(numSamples - dataIdx) < batchSize)
			        ? (numSamples - dataIdx)
			        : static_cast<unsigned int>(batchSize);

			// === Train Discriminator A (parallel) ===
			for (int critic = 0; critic < config.nCriticPerGenerator; ++critic)
			{
				float batchDLossA = 0.0f;
				float batchInfoLossA = 0.0f;

				// Phase 1: Pre-generate data (sequential for RNG determinism)
				preRealPtrsA.resize(curBatchSize);
				preRealPtrsB.resize(curBatchSize);
				preRealSizesA.resize(curBatchSize);
				preRealSizesB.resize(curBatchSize);
				for (unsigned int s = 0; s < curBatchSize; ++s)
				{
					const unsigned int idxA = (dataIdx + s) % sizeA;
					const unsigned int idxB = (dataIdx + s) % sizeB;
					domainA->getTrainRowView(idxA, preRealPtrsA[s], preRealSizesA[s]);
					domainB->getTrainRowView(idxB, preRealPtrsB[s], preRealSizesB[s]);
				}
				if (hasInfo) { preCatCode.resize(curBatchSize); preContCode.resize(curBatchSize); }
				if (hasStyle) preStyleNoise.resize(curBatchSize);
				if (config.lossType == GANConfig::GAN_WGAN_GP) preGPEpsilon.resize(curBatchSize);
				for (unsigned int s = 0; s < curBatchSize; ++s)
				{
					if (hasInfo) sampleLatentCodes(preCatCode[s], preContCode[s]);
					if (hasStyle) sampleNoise(preStyleNoise[s]);
					if (config.lossType == GANConfig::GAN_WGAN_GP)
						preGPEpsilon[s] = glades::rng::unit_float01(rngEngine);
				}

				// Phase 2: Zero thread contexts
				for (unsigned int tid = 0; tid < nThreads; ++tid)
				{
					threadCtx[tid].zeroGrads();
					threadCtx[tid].zeroLosses();
				}

				// Phase 3: Parallel compute
				beginToTid.assign(curBatchSize, 0u);
				for (unsigned int id = 0; id < nThreads; ++id)
				{
					const unsigned int b = id * curBatchSize / nThreads;
					beginToTid[b] = id;
				}

				CycleDiscData dcdA;
				dcdA.self = this;
				dcdA.ctxs = &threadCtx[0];
				dcdA.nThreads = nThreads;
				dcdA.curBatchSize = curBatchSize;
				dcdA.realPtrsA = &preRealPtrsA[0];
				dcdA.realPtrsB = &preRealPtrsB[0];
				dcdA.catCodes = hasInfo ? &preCatCode[0] : NULL;
				dcdA.contCodes = hasInfo ? &preContCode[0] : NULL;
				dcdA.styleNoise = hasStyle ? &preStyleNoise[0] : NULL;
				dcdA.gpEps = (config.lossType == GANConfig::GAN_WGAN_GP) ? &preGPEpsilon[0] : NULL;
				dcdA.genInputDim = genBAInputDim;
				dcdA.srcDim = dimB;
				dcdA.tgtDim = dimA;
				dcdA.numCat = numCat;
				dcdA.numCont = numCont;
				dcdA.hasInfo = hasInfo;
				dcdA.hasStyle = hasStyle;
				dcdA.genIsDeconv = genIsDeconv;
				dcdA.infoLambda = infoLambda;
				dcdA.penultDim = penultDimA;
				dcdA.beginToTid = &beginToTid[0];
				dcdA.isDiscA = true;
				ThreadPool::instance().parallel_for(curBatchSize, cycleDiscBody, &dcdA);

				// Phase 4: Ordered reduction
				if (config.archType == GANConfig::GAN_DFF)
					zeroDFFGrads(discriminator);
				else
					zeroCNNGrads(discriminator);
				if (hasInfo)
				{
					std::memset(&qHead.gW[0], 0, qHead.gW.size() * sizeof(float));
					std::memset(&qHead.gBias[0], 0, qHead.gBias.size() * sizeof(float));
				}

				for (unsigned int tid = 0; tid < nThreads; ++tid)
				{
					if (config.archType == GANConfig::GAN_DFF)
						threadCtx[tid].discGrads.addToDFF(discriminator);
					else
						threadCtx[tid].discGrads.addToCNN(discriminator);
					if (hasInfo)
					{
						for (size_t i = 0; i < qHead.gW.size(); ++i)
							qHead.gW[i] += threadCtx[tid].qGrads.qGW[i];
						for (size_t j = 0; j < qHead.gBias.size(); ++j)
							qHead.gBias[j] += threadCtx[tid].qGrads.qGBias[j];
					}
					batchDLossA += threadCtx[tid].dLossReal;
					batchInfoLossA += threadCtx[tid].infoLoss;
					epochWasserstein += threadCtx[tid].wasserstein;
					epochCatCorrect += threadCtx[tid].catCorrect;
					epochCatTotal += threadCtx[tid].catTotal;
				}

				const float invBatchA = 1.0f / static_cast<float>(curBatchSize);
				epochDLossA += batchDLossA * invBatchA;
				if (hasInfo) epochInfoLoss += batchInfoLossA * invBatchA;

				// Adaptive balance: skip disc A update when disc is dominant
				const float avgDiscLossA = batchDLossA * invBatchA;
				const bool skipDiscAUpdate = config.adaptiveBalance &&
				    (avgDiscLossA < config.adaptiveDiscThreshold);
				if (!skipDiscAUpdate)
				{
					if (config.archType == GANConfig::GAN_DFF)
					{
						for (unsigned int t = 0; t < discriminator.tensorDff.T.size(); ++t)
						{
							NNetwork::TensorDFFState::Transition& tr = discriminator.tensorDff.T[t];
							for (size_t i = 0; i < tr.gW.size(); ++i) tr.gW[i] *= invBatchA;
							for (size_t j = 0; j < tr.gBias.size(); ++j) tr.gBias[j] *= invBatchA;
						}
						dffUpdate(discriminator, discAdamState, config.discriminatorLR);
					}
					else
					{
						for (unsigned int l = 0; l < discriminator.tensorCnn.convLayers.size(); ++l)
						{
							NNetwork::TensorCNNState::ConvLayer& cl = discriminator.tensorCnn.convLayers[l];
							for (size_t i = 0; i < cl.gW.size(); ++i) cl.gW[i] *= invBatchA;
							for (size_t j = 0; j < cl.gBias.size(); ++j) cl.gBias[j] *= invBatchA;
						}
						for (unsigned int t = 0; t < discriminator.tensorCnn.fcLayers.size(); ++t)
						{
							NNetwork::TensorCNNState::FCTransition& fc = discriminator.tensorCnn.fcLayers[t];
							for (size_t i = 0; i < fc.gW.size(); ++i) fc.gW[i] *= invBatchA;
							for (size_t j = 0; j < fc.gBias.size(); ++j) fc.gBias[j] *= invBatchA;
						}
						cnnUpdate(discriminator, discAdamState, config.discriminatorLR);
					}

					// Spectral normalization on disc A
					if (config.spectralNorm)
					{
						if (config.archType == GANConfig::GAN_DFF)
							spectralNormDFF(discriminator, discSNState, config.spectralNormIters);
						else
							spectralNormCNN(discriminator, discSNState, config.spectralNormIters);
					}

					if (hasInfo)
					{
						for (size_t i = 0; i < qHead.gW.size(); ++i) qHead.gW[i] *= invBatchA;
						for (unsigned int j = 0; j < qHead.qOutDim; ++j) qHead.gBias[j] *= invBatchA;
						qHeadUpdate(qHead, qAdamState, config.discriminatorLR);
					}
				}
			}

			// === Train Discriminator B (parallel) ===
			for (int critic = 0; critic < config.nCriticPerGenerator; ++critic)
			{
				float batchDLossB = 0.0f;
				float batchInfoLossB = 0.0f;

				// Phase 1: Pre-generate data
				preRealPtrsA.resize(curBatchSize);
				preRealPtrsB.resize(curBatchSize);
				preRealSizesA.resize(curBatchSize);
				preRealSizesB.resize(curBatchSize);
				for (unsigned int s = 0; s < curBatchSize; ++s)
				{
					const unsigned int idxA = (dataIdx + s) % sizeA;
					const unsigned int idxB = (dataIdx + s) % sizeB;
					domainA->getTrainRowView(idxA, preRealPtrsA[s], preRealSizesA[s]);
					domainB->getTrainRowView(idxB, preRealPtrsB[s], preRealSizesB[s]);
				}
				if (hasInfo) { preCatCode.resize(curBatchSize); preContCode.resize(curBatchSize); }
				if (hasStyle) preStyleNoise.resize(curBatchSize);
				if (config.lossType == GANConfig::GAN_WGAN_GP) preGPEpsilon.resize(curBatchSize);
				for (unsigned int s = 0; s < curBatchSize; ++s)
				{
					if (hasInfo) sampleLatentCodes(preCatCode[s], preContCode[s]);
					if (hasStyle) sampleNoise(preStyleNoise[s]);
					if (config.lossType == GANConfig::GAN_WGAN_GP)
						preGPEpsilon[s] = glades::rng::unit_float01(rngEngine);
				}

				// Phase 2: Zero thread contexts
				for (unsigned int tid = 0; tid < nThreads; ++tid)
				{
					threadCtx[tid].zeroGrads();
					threadCtx[tid].zeroLosses();
				}

				// Phase 3: Parallel compute
				beginToTid.assign(curBatchSize, 0u);
				for (unsigned int id = 0; id < nThreads; ++id)
				{
					const unsigned int b = id * curBatchSize / nThreads;
					beginToTid[b] = id;
				}

				CycleDiscData dcdB;
				dcdB.self = this;
				dcdB.ctxs = &threadCtx[0];
				dcdB.nThreads = nThreads;
				dcdB.curBatchSize = curBatchSize;
				dcdB.realPtrsA = &preRealPtrsA[0];
				dcdB.realPtrsB = &preRealPtrsB[0];
				dcdB.catCodes = hasInfo ? &preCatCode[0] : NULL;
				dcdB.contCodes = hasInfo ? &preContCode[0] : NULL;
				dcdB.styleNoise = hasStyle ? &preStyleNoise[0] : NULL;
				dcdB.gpEps = (config.lossType == GANConfig::GAN_WGAN_GP) ? &preGPEpsilon[0] : NULL;
				dcdB.genInputDim = genABInputDim;
				dcdB.srcDim = dimA;
				dcdB.tgtDim = dimB;
				dcdB.numCat = numCat;
				dcdB.numCont = numCont;
				dcdB.hasInfo = hasInfo;
				dcdB.hasStyle = hasStyle;
				dcdB.genIsDeconv = genIsDeconv;
				dcdB.infoLambda = infoLambda;
				dcdB.penultDim = penultDimB;
				dcdB.beginToTid = &beginToTid[0];
				dcdB.isDiscA = false;
				ThreadPool::instance().parallel_for(curBatchSize, cycleDiscBody, &dcdB);

				// Phase 4: Ordered reduction
				if (config.archType == GANConfig::GAN_DFF)
					zeroDFFGrads(discriminatorB);
				else
					zeroCNNGrads(discriminatorB);
				if (hasInfo)
				{
					std::memset(&qHeadB.gW[0], 0, qHeadB.gW.size() * sizeof(float));
					std::memset(&qHeadB.gBias[0], 0, qHeadB.gBias.size() * sizeof(float));
				}

				for (unsigned int tid = 0; tid < nThreads; ++tid)
				{
					if (config.archType == GANConfig::GAN_DFF)
						threadCtx[tid].discBGrads.addToDFF(discriminatorB);
					else
						threadCtx[tid].discBGrads.addToCNN(discriminatorB);
					if (hasInfo)
					{
						for (size_t i = 0; i < qHeadB.gW.size(); ++i)
							qHeadB.gW[i] += threadCtx[tid].qBGrads.qGW[i];
						for (size_t j = 0; j < qHeadB.gBias.size(); ++j)
							qHeadB.gBias[j] += threadCtx[tid].qBGrads.qGBias[j];
					}
					batchDLossB += threadCtx[tid].dLossReal;
					batchInfoLossB += threadCtx[tid].infoLoss;
					epochWasserstein += threadCtx[tid].wasserstein;
					epochCatCorrect += threadCtx[tid].catCorrect;
					epochCatTotal += threadCtx[tid].catTotal;
				}

				const float invBatchB = 1.0f / static_cast<float>(curBatchSize);
				epochDLossB += batchDLossB * invBatchB;
				if (hasInfo) epochInfoLoss += batchInfoLossB * invBatchB;

				// Adaptive balance: skip disc B update when disc is dominant
				const float avgDiscLossB = batchDLossB * invBatchB;
				const bool skipDiscBUpdate = config.adaptiveBalance &&
				    (avgDiscLossB < config.adaptiveDiscThreshold);
				if (!skipDiscBUpdate)
				{
					if (config.archType == GANConfig::GAN_DFF)
					{
						for (unsigned int t = 0; t < discriminatorB.tensorDff.T.size(); ++t)
						{
							NNetwork::TensorDFFState::Transition& tr = discriminatorB.tensorDff.T[t];
							for (size_t i = 0; i < tr.gW.size(); ++i) tr.gW[i] *= invBatchB;
							for (size_t j = 0; j < tr.gBias.size(); ++j) tr.gBias[j] *= invBatchB;
						}
						dffUpdate(discriminatorB, discBAdamState, config.discriminatorLR);
					}
					else
					{
						for (unsigned int l = 0; l < discriminatorB.tensorCnn.convLayers.size(); ++l)
						{
							NNetwork::TensorCNNState::ConvLayer& cl = discriminatorB.tensorCnn.convLayers[l];
							for (size_t i = 0; i < cl.gW.size(); ++i) cl.gW[i] *= invBatchB;
							for (size_t j = 0; j < cl.gBias.size(); ++j) cl.gBias[j] *= invBatchB;
						}
						for (unsigned int t = 0; t < discriminatorB.tensorCnn.fcLayers.size(); ++t)
						{
							NNetwork::TensorCNNState::FCTransition& fc = discriminatorB.tensorCnn.fcLayers[t];
							for (size_t i = 0; i < fc.gW.size(); ++i) fc.gW[i] *= invBatchB;
							for (size_t j = 0; j < fc.gBias.size(); ++j) fc.gBias[j] *= invBatchB;
						}
						cnnUpdate(discriminatorB, discBAdamState, config.discriminatorLR);
					}

					// Spectral normalization on disc B
					if (config.spectralNorm)
					{
						if (config.archType == GANConfig::GAN_DFF)
							spectralNormDFF(discriminatorB, discBSNState, config.spectralNormIters);
						else
							spectralNormCNN(discriminatorB, discBSNState, config.spectralNormIters);
					}

					if (hasInfo)
					{
						for (size_t i = 0; i < qHeadB.gW.size(); ++i) qHeadB.gW[i] *= invBatchB;
						for (unsigned int j = 0; j < qHeadB.qOutDim; ++j) qHeadB.gBias[j] *= invBatchB;
						qHeadUpdate(qHeadB, qBAdamState, config.discriminatorLR);
					}
				}
			}

			// === Generator step (both G_AB and G_BA) (parallel) ===
			for (int gstep = 0; gstep < config.nGenPerCritic; ++gstep)
			{
				float batchGLossAB = 0.0f;
				float batchGLossBA = 0.0f;
				float batchCycleLoss = 0.0f;
				float batchIdentityLoss = 0.0f;
				float batchInfoLoss = 0.0f;

				// Phase 1: Pre-generate data
				preRealPtrsA.resize(curBatchSize);
				preRealPtrsB.resize(curBatchSize);
				for (unsigned int s = 0; s < curBatchSize; ++s)
				{
					const unsigned int idxA = (dataIdx + s) % sizeA;
					const unsigned int idxB = (dataIdx + s) % sizeB;
					domainA->getTrainRowView(idxA, preRealPtrsA[s], preRealSizesA[s]);
					domainB->getTrainRowView(idxB, preRealPtrsB[s], preRealSizesB[s]);
				}
				if (hasInfo)
				{
					preCatCodeAB.resize(curBatchSize); preContCodeAB.resize(curBatchSize);
					preCatCodeBA.resize(curBatchSize); preContCodeBA.resize(curBatchSize);
				}
				if (hasStyle)
				{
					preStyleNoiseAB.resize(curBatchSize);
					preStyleNoiseBA.resize(curBatchSize);
					preStyleNoiseRecA.resize(curBatchSize);
					preStyleNoiseRecB.resize(curBatchSize);
					preStyleNoiseIdentB.resize(curBatchSize);
					preStyleNoiseIdentA.resize(curBatchSize);
				}
				for (unsigned int s = 0; s < curBatchSize; ++s)
				{
					if (hasInfo)
					{
						sampleLatentCodes(preCatCodeAB[s], preContCodeAB[s]);
						sampleLatentCodes(preCatCodeBA[s], preContCodeBA[s]);
					}
					if (hasStyle)
					{
						sampleNoise(preStyleNoiseAB[s]);
						sampleNoise(preStyleNoiseBA[s]);
						sampleNoise(preStyleNoiseRecA[s]);
						sampleNoise(preStyleNoiseRecB[s]);
						sampleNoise(preStyleNoiseIdentB[s]);
						sampleNoise(preStyleNoiseIdentA[s]);
					}
				}

				// Phase 2: Zero thread contexts
				for (unsigned int tid = 0; tid < nThreads; ++tid)
				{
					threadCtx[tid].zeroGrads();
					threadCtx[tid].zeroLosses();
				}
				// Refresh per-thread LN scratch gamma/beta
				if (hasLN)
				{
					for (unsigned int tid = 0; tid < nThreads; ++tid)
					{
						genLNScratch[tid].gamma = genLNParams.gamma;
						genLNScratch[tid].beta = genLNParams.beta;
					}
				}
				if (hasBALN)
				{
					for (unsigned int tid = 0; tid < nThreads; ++tid)
					{
						genBALNScratch[tid].gamma = genBALNParams.gamma;
						genBALNScratch[tid].beta = genBALNParams.beta;
					}
				}

				// Phase 3: Parallel compute
				beginToTid.assign(curBatchSize, 0u);
				for (unsigned int id = 0; id < nThreads; ++id)
				{
					const unsigned int b = id * curBatchSize / nThreads;
					beginToTid[b] = id;
				}

				CycleGenData cgd;
				cgd.self = this;
				cgd.ctxs = &threadCtx[0];
				cgd.nThreads = nThreads;
				cgd.curBatchSize = curBatchSize;
				cgd.realPtrsA = &preRealPtrsA[0];
				cgd.realPtrsB = &preRealPtrsB[0];
				cgd.catCodesAB = hasInfo ? &preCatCodeAB[0] : NULL;
				cgd.contCodesAB = hasInfo ? &preContCodeAB[0] : NULL;
				cgd.catCodesBA = hasInfo ? &preCatCodeBA[0] : NULL;
				cgd.contCodesBA = hasInfo ? &preContCodeBA[0] : NULL;
				cgd.styleNoiseAB = hasStyle ? &preStyleNoiseAB[0] : NULL;
				cgd.styleNoiseBA = hasStyle ? &preStyleNoiseBA[0] : NULL;
				cgd.styleNoiseRecA = hasStyle ? &preStyleNoiseRecA[0] : NULL;
				cgd.styleNoiseRecB = hasStyle ? &preStyleNoiseRecB[0] : NULL;
				cgd.styleNoiseIdentB = hasStyle ? &preStyleNoiseIdentB[0] : NULL;
				cgd.styleNoiseIdentA = hasStyle ? &preStyleNoiseIdentA[0] : NULL;
				cgd.genABInputDim = genABInputDim;
				cgd.genBAInputDim = genBAInputDim;
				cgd.dimA = dimA;
				cgd.dimB = dimB;
				cgd.numCat = numCat;
				cgd.numCont = numCont;
				cgd.hasInfo = hasInfo;
				cgd.hasStyle = hasStyle;
				cgd.genIsDeconv = genIsDeconv;
				cgd.hasLN = hasLN;
				cgd.hasBALN = hasBALN;
				cgd.cycleLambda = cycleLambda;
				cgd.identityLambda = identityLambda;
				cgd.infoLambda = infoLambda;
				cgd.penultDimA = penultDimA;
				cgd.penultDimB = penultDimB;
				cgd.beginToTid = &beginToTid[0];
				cgd.lnScratchAB = hasLN ? &genLNScratch[0] : NULL;
				cgd.lnScratchBA = hasBALN ? &genBALNScratch[0] : NULL;
				ThreadPool::instance().parallel_for(curBatchSize, cycleGenBody, &cgd);

				// Phase 4: Ordered reduction
				// Zero master grads for G_AB
				if (genIsDeconv)
					zeroDeconvGrads(generator);
				else
					zeroDFFGrads(generator);
				// Zero master grads for G_BA
				if (genIsDeconv)
					zeroDeconvGrads(generatorBA);
				else
					zeroDFFGrads(generatorBA);
				// Zero style apparatus if active
				if (hasStyle && !genIsDeconv)
				{
					zeroDFFGrads(mappingNet);
					for (size_t sa = 0; sa < styleAffines.size(); ++sa)
					{
						std::memset(&styleAffines[sa].gW[0], 0, styleAffines[sa].gW.size() * sizeof(float));
						std::memset(&styleAffines[sa].gBias[0], 0, styleAffines[sa].gBias.size() * sizeof(float));
						gNoiseScales[sa] = 0.0f;
					}
					zeroDFFGrads(mappingNetBA);
					for (size_t sa = 0; sa < styleAffinesBA.size(); ++sa)
					{
						std::memset(&styleAffinesBA[sa].gW[0], 0, styleAffinesBA[sa].gW.size() * sizeof(float));
						std::memset(&styleAffinesBA[sa].gBias[0], 0, styleAffinesBA[sa].gBias.size() * sizeof(float));
						gNoiseScalesBA[sa] = 0.0f;
					}
				}
				if (hasLN)
				{
					for (size_t i = 0; i < genLNParams.gGamma.size(); ++i)
					{
						std::memset(&genLNParams.gGamma[i][0], 0, genLNParams.gGamma[i].size() * sizeof(float));
						std::memset(&genLNParams.gBeta[i][0], 0, genLNParams.gBeta[i].size() * sizeof(float));
					}
				}
				if (hasBALN)
				{
					for (size_t i = 0; i < genBALNParams.gGamma.size(); ++i)
					{
						std::memset(&genBALNParams.gGamma[i][0], 0, genBALNParams.gGamma[i].size() * sizeof(float));
						std::memset(&genBALNParams.gBeta[i][0], 0, genBALNParams.gBeta[i].size() * sizeof(float));
					}
				}

				// Sum thread-local grads (in order for determinism)
				for (unsigned int tid = 0; tid < nThreads; ++tid)
				{
					// G_AB grads
					if (genIsDeconv)
						threadCtx[tid].genGrads.addToDeconv(generator);
					else
						threadCtx[tid].genGrads.addToDFF(generator);
					// G_BA grads
					if (genIsDeconv)
						threadCtx[tid].genBAGrads.addToDeconv(generatorBA);
					else
						threadCtx[tid].genBAGrads.addToDFF(generatorBA);
					// Style AB
					if (hasStyle && !genIsDeconv)
					{
						threadCtx[tid].mappingGrads.addToDFF(mappingNet);
						for (size_t sa = 0; sa < styleAffines.size(); ++sa)
						{
							for (size_t i = 0; i < styleAffines[sa].gW.size(); ++i)
								styleAffines[sa].gW[i] += threadCtx[tid].styleGW[sa][i];
							for (size_t j = 0; j < styleAffines[sa].gBias.size(); ++j)
								styleAffines[sa].gBias[j] += threadCtx[tid].styleGBias[sa][j];
							gNoiseScales[sa] += threadCtx[tid].gNoiseScalesLocal[sa];
						}
						// Style BA
						threadCtx[tid].mappingBAGrads.addToDFF(mappingNetBA);
						for (size_t sa = 0; sa < styleAffinesBA.size(); ++sa)
						{
							for (size_t i = 0; i < styleAffinesBA[sa].gW.size(); ++i)
								styleAffinesBA[sa].gW[i] += threadCtx[tid].styleBAGW[sa][i];
							for (size_t j = 0; j < styleAffinesBA[sa].gBias.size(); ++j)
								styleAffinesBA[sa].gBias[j] += threadCtx[tid].styleBAGBias[sa][j];
							gNoiseScalesBA[sa] += threadCtx[tid].gNoiseScalesBALocal[sa];
						}
					}
					// LN AB
					if (hasLN)
					{
						for (size_t i = 0; i < genLNParams.gGamma.size(); ++i)
						{
							for (size_t j = 0; j < genLNParams.gGamma[i].size(); ++j)
								genLNParams.gGamma[i][j] += threadCtx[tid].lnGGamma[i][j];
							for (size_t j = 0; j < genLNParams.gBeta[i].size(); ++j)
								genLNParams.gBeta[i][j] += threadCtx[tid].lnGBeta[i][j];
						}
					}
					// LN BA
					if (hasBALN)
					{
						for (size_t i = 0; i < genBALNParams.gGamma.size(); ++i)
						{
							for (size_t j = 0; j < genBALNParams.gGamma[i].size(); ++j)
								genBALNParams.gGamma[i][j] += threadCtx[tid].lnBAGGamma[i][j];
							for (size_t j = 0; j < genBALNParams.gBeta[i].size(); ++j)
								genBALNParams.gBeta[i][j] += threadCtx[tid].lnBAGBeta[i][j];
						}
					}

					batchGLossAB += threadCtx[tid].gLossAB;
					batchGLossBA += threadCtx[tid].gLossBA;
					batchCycleLoss += threadCtx[tid].cycleLoss;
					batchIdentityLoss += threadCtx[tid].identityLoss;
					batchInfoLoss += threadCtx[tid].infoLoss;
				}

				const float invBatch = 1.0f / static_cast<float>(curBatchSize);
				epochGLossAB += batchGLossAB * invBatch;
				epochGLossBA += batchGLossBA * invBatch;
				epochCycleLoss += batchCycleLoss * invBatch;
				epochIdentityLoss += batchIdentityLoss * invBatch;

				// Scale and update G_AB
				if (genIsDeconv)
				{
					NNetwork::TensorDeconvState& ds = generator.tensorDeconv;
					for (size_t i = 0; i < ds.fcGW.size(); ++i) ds.fcGW[i] *= invBatch;
					for (unsigned int j = 0; j < ds.fcOut; ++j) ds.fcGBias[j] *= invBatch;
					for (unsigned int l = 0; l < ds.layers.size(); ++l)
					{
						NNetwork::TensorDeconvState::DeconvLayer& dl = ds.layers[l];
						for (size_t i = 0; i < dl.gW.size(); ++i) dl.gW[i] *= invBatch;
						for (unsigned int c = 0; c < dl.outC; ++c) dl.gBias[c] *= invBatch;
						if (dl.useBatchNorm)
						{
							for (unsigned int c = 0; c < dl.outC; ++c) dl.gBnGamma[c] *= invBatch;
							for (unsigned int c = 0; c < dl.outC; ++c) dl.gBnBeta[c] *= invBatch;
						}
					}
					deconvUpdate(generator, config.generatorLR);
				}
				else
				{
					for (unsigned int t = 0; t < generator.tensorDff.T.size(); ++t)
					{
						NNetwork::TensorDFFState::Transition& tr = generator.tensorDff.T[t];
						for (size_t i = 0; i < tr.gW.size(); ++i) tr.gW[i] *= invBatch;
						for (size_t j = 0; j < tr.gBias.size(); ++j) tr.gBias[j] *= invBatch;
					}
					dffUpdate(generator, genAdamState, config.generatorLR);
					if (hasLN)
					{
						scaleLayerNormGrads(genLNParams, invBatch);
						layerNormUpdate(genLNParams, config.generatorLR);
					}
				}

				// Scale and update G_BA
				if (genIsDeconv)
				{
					NNetwork::TensorDeconvState& ds = generatorBA.tensorDeconv;
					for (size_t i = 0; i < ds.fcGW.size(); ++i) ds.fcGW[i] *= invBatch;
					for (unsigned int j = 0; j < ds.fcOut; ++j) ds.fcGBias[j] *= invBatch;
					for (unsigned int l = 0; l < ds.layers.size(); ++l)
					{
						NNetwork::TensorDeconvState::DeconvLayer& dl = ds.layers[l];
						for (size_t i = 0; i < dl.gW.size(); ++i) dl.gW[i] *= invBatch;
						for (unsigned int c = 0; c < dl.outC; ++c) dl.gBias[c] *= invBatch;
						if (dl.useBatchNorm)
						{
							for (unsigned int c = 0; c < dl.outC; ++c) dl.gBnGamma[c] *= invBatch;
							for (unsigned int c = 0; c < dl.outC; ++c) dl.gBnBeta[c] *= invBatch;
						}
					}
					deconvUpdate(generatorBA, config.generatorLR);
				}
				else
				{
					for (unsigned int t = 0; t < generatorBA.tensorDff.T.size(); ++t)
					{
						NNetwork::TensorDFFState::Transition& tr = generatorBA.tensorDff.T[t];
						for (size_t i = 0; i < tr.gW.size(); ++i) tr.gW[i] *= invBatch;
						for (size_t j = 0; j < tr.gBias.size(); ++j) tr.gBias[j] *= invBatch;
					}
					dffUpdate(generatorBA, genBAAdamState, config.generatorLR);
					if (hasBALN)
					{
						scaleLayerNormGrads(genBALNParams, invBatch);
						layerNormUpdate(genBALNParams, config.generatorLR);
					}
				}

				// Scale and update style apparatus if active
				if (hasStyle)
				{
					for (unsigned int t = 0; t < mappingNet.tensorDff.T.size(); ++t)
					{
						NNetwork::TensorDFFState::Transition& tr = mappingNet.tensorDff.T[t];
						for (size_t i = 0; i < tr.gW.size(); ++i) tr.gW[i] *= invBatch;
						for (size_t j = 0; j < tr.gBias.size(); ++j) tr.gBias[j] *= invBatch;
					}
					dffUpdate(mappingNet, mappingAdamState, config.generatorLR);
					for (unsigned int t = 0; t < styleAffines.size(); ++t)
					{
						for (size_t i = 0; i < styleAffines[t].gW.size(); ++i) styleAffines[t].gW[i] *= invBatch;
						for (unsigned int j = 0; j < styleAffines[t].outDim; ++j) styleAffines[t].gBias[j] *= invBatch;
						gNoiseScales[t] *= invBatch;
					}
					styleAffineUpdate(config.generatorLR, styleAffines, styleAffineAdamState, noiseScales, gNoiseScales);

					for (unsigned int t = 0; t < mappingNetBA.tensorDff.T.size(); ++t)
					{
						NNetwork::TensorDFFState::Transition& tr = mappingNetBA.tensorDff.T[t];
						for (size_t i = 0; i < tr.gW.size(); ++i) tr.gW[i] *= invBatch;
						for (size_t j = 0; j < tr.gBias.size(); ++j) tr.gBias[j] *= invBatch;
					}
					dffUpdate(mappingNetBA, mappingBAAdamState, config.generatorLR);
					for (unsigned int t = 0; t < styleAffinesBA.size(); ++t)
					{
						for (size_t i = 0; i < styleAffinesBA[t].gW.size(); ++i) styleAffinesBA[t].gW[i] *= invBatch;
						for (unsigned int j = 0; j < styleAffinesBA[t].outDim; ++j) styleAffinesBA[t].gBias[j] *= invBatch;
						gNoiseScalesBA[t] *= invBatch;
					}
					styleAffineUpdate(config.generatorLR, styleAffinesBA, styleAffineBAAdamState, noiseScalesBA, gNoiseScalesBA);
				}
			}

			dataIdx += curBatchSize;
		}

		const float invBatches = 1.0f / static_cast<float>(numBatches);
		GANEpochMetrics metrics;
		metrics.epoch = epoch;
		metrics.dLossA = epochDLossA * invBatches;
		metrics.dLossB = epochDLossB * invBatches;
		metrics.gLossAB = epochGLossAB * invBatches;
		metrics.gLossBA = epochGLossBA * invBatches;
		metrics.cycleLoss = epochCycleLoss * invBatches;
		metrics.identityLoss = epochIdentityLoss * invBatches;
		metrics.gLoss = (epochGLossAB + epochGLossBA) * 0.5f * invBatches;
		if (config.lossType == GANConfig::GAN_WGAN_GP)
		{
			const unsigned int epochSamples = numSamples * static_cast<unsigned int>(config.nCriticPerGenerator);
			if (epochSamples > 0u)
				metrics.wasserstein = epochWasserstein / static_cast<float>(epochSamples);
		}
		if (hasInfo)
		{
			metrics.infoLoss = epochInfoLoss * invBatches;
			if (epochCatTotal > 0u)
				metrics.catAccuracy = static_cast<float>(epochCatCorrect) / static_cast<float>(epochCatTotal);
		}

		if (cb)
		{
			cb->onEpochEnd(metrics);
			if (cb->shouldStop(metrics))
				break;
		}
	}

	return NNetworkStatus();
}

// ============================================================
// CycleGAN: Domain translation
// ============================================================

NNetworkStatus GAN::translate(const DataInput* input, bool aToB,
                              std::vector<std::vector<float> >& outSamples) const
{
	return translate(input, aToB, NULL, NULL, outSamples);
}

NNetworkStatus GAN::translate(const DataInput* input, bool aToB,
                              const std::vector<float>* fixedCatCode,
                              const std::vector<float>* fixedContCode,
                              std::vector<std::vector<float> >& outSamples) const
{
	const bool hasInfo = config.useInfo || (config.variantType == GANConfig::GAN_INFO);
	const bool hasStyle = config.useStyle || (config.variantType == GANConfig::GAN_STYLE);

	const NNetwork& gen = aToB ? generator : generatorBA;
	LayerNormParams* lnp = aToB ? genLN() : genBALN();
	if (!gen.tensorDff.initialized && !gen.tensorDeconv.initialized)
		return NNetworkStatus(NNetworkStatus::INVALID_STATE, "GAN::translate: generator not trained");

	const unsigned int n = input->getTrainSize();
	if (n == 0u)
		return NNetworkStatus(NNetworkStatus::EMPTY_DATA, "GAN::translate: no input data");

	const unsigned int numCat = hasInfo ? config.infoConfig.numCategorical : 0u;
	const unsigned int numCont = hasInfo ? config.infoConfig.numContinuous : 0u;

	// Select mapping net and style affines for direction
	const NNetwork& mapNet = aToB ? mappingNet : mappingNetBA;
	const std::vector<StyleAffine>& affines = aToB ? styleAffines : styleAffinesBA;
	const std::vector<float>& scales = aToB ? noiseScales : noiseScalesBA;

	outSamples.resize(n);
	for (unsigned int s = 0; s < n; ++s)
	{
		const float* row = NULL;
		unsigned int dim = 0u;
		input->getTrainRowView(s, row, dim);

		// Build input [domain_data | codes?]
		const unsigned int inputDim = dim + numCat + numCont;
		std::vector<float> genInput(inputDim);
		std::memcpy(&genInput[0], row, dim * sizeof(float));

		if (hasInfo)
		{
			std::vector<float> catCode, contCode;
			if (fixedCatCode)
				catCode = *fixedCatCode;
			else
				sampleLatentCodes(catCode, contCode);
			if (fixedContCode)
				contCode = *fixedContCode;
			else if (!fixedCatCode)
			{
				// already sampled above
			}
			else
				sampleLatentCodes(catCode, contCode);

			catCode.resize(numCat, 0.0f);
			contCode.resize(numCont, 0.0f);

			if (numCat > 0u) std::memcpy(&genInput[dim], &catCode[0], numCat * sizeof(float));
			if (numCont > 0u) std::memcpy(&genInput[dim + numCat], &contCode[0], numCont * sizeof(float));
		}

		const bool isDeconv = gen.tensorDeconv.initialized;

		if (hasStyle && mapNet.tensorDff.initialized)
		{
			std::vector<float> noise;
			sampleNoise(noise);
			std::vector<std::vector<float> > mapAct;
			dffForward(mapNet, &noise[0], config.noiseDim, mapAct);
			const std::vector<float>& w = mapAct.back();

			std::vector<std::vector<float> > genAct, xN;
			std::vector<float> gM, gI;
			std::vector<std::vector<float> > gNV;
			dffForwardStyled(gen, &genInput[0], inputDim, w,
			                 affines, scales,
			                 genAct, xN, gM, gI, gNV);
			outSamples[s] = genAct.back();
		}
		else if (isDeconv)
		{
			std::vector<float> output;
			deconvForward(gen, &genInput[0], inputDim, output);
			outSamples[s] = output;
		}
		else
		{
			std::vector<std::vector<float> > genAct;
			dffForward(gen, &genInput[0], inputDim, genAct, config.generatorOutputSigmoid, lnp);
			outSamples[s] = genAct.back();
		}
	}

	return NNetworkStatus();
}
