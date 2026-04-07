// GPU transformer state implementation.
#include "gpu_transformer_state.h"
#include "gpu_kernels.h"

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

namespace glades {
namespace gpu {

// ---- GpuTransformerWeights ----

GpuTransformerWeights::GpuTransformerWeights()
    : initialized(false),
      dModel(0), dFF(0), nHeads(0), nKVHeads(0), nLayers(0),
      vocabSize(0), inputSize(0), outSize(0), ffnKind(0),
      tokenModel(false), tieEmbeddings(false),
      blocks(0),
      d_adamParams(0), d_adamGrads(0), d_adamM(0), d_adamV(0),
      d_adamLr(0), d_adamWd(0), d_adamSizes(0),
      adamGroupCount(0), adamMaxSize(0), adamPtrsUploaded(false)
{
}

GpuTransformerWeights::~GpuTransformerWeights()
{
	free();
}

static bool allocBuf(GpuBuffer<float>& buf, size_t n)
{
	if (n == 0)
		return true;
	return buf.allocate(n);
}

bool GpuTransformerWeights::allocate(unsigned int dm, unsigned int df, unsigned int nh,
                                      unsigned int nkvh, unsigned int nl,
                                      unsigned int vs, unsigned int is, unsigned int os,
                                      unsigned int ffk, bool tm, bool te,
                                      bool skipAdamBufs)
{
	free();

	dModel = dm;
	dFF = df;
	nHeads = nh;
	nKVHeads = nkvh;
	nLayers = nl;
	vocabSize = vs;
	inputSize = is;
	outSize = os;
	ffnKind = ffk;
	tokenModel = tm;
	tieEmbeddings = te;

	const unsigned int dHead = dModel / nHeads;
	const unsigned int dModelKV = nKVHeads * dHead;
	const unsigned int ff1Width = (ffnKind == 1) ? (2u * dFF) : dFF; // SwiGLU vs MLP

	// Token embedding
	if (tokenModel)
	{
		if (!allocBuf(tokE, (size_t)vs * dm)) return false;
		if (!skipAdamBufs && !allocBuf(vTokE, (size_t)vs * dm)) return false;
		if (!skipAdamBufs && !allocBuf(v2TokE, (size_t)vs * dm)) return false;
		if (!allocBuf(gTokE, (size_t)vs * dm)) return false;
		if (!allocBuf(lmBias, vs)) return false;
		if (!skipAdamBufs && !allocBuf(mLmBias, vs)) return false;
		if (!skipAdamBufs && !allocBuf(v2LmBias, vs)) return false;
		if (!allocBuf(gLmBias, vs)) return false;
	}

	// Input projection
	if (!tokenModel)
	{
		if (!allocBuf(WIn, (size_t)dm * is)) return false;
		if (!skipAdamBufs && !allocBuf(vWIn, (size_t)dm * is)) return false;
		if (!skipAdamBufs && !allocBuf(v2WIn, (size_t)dm * is)) return false;
		if (!allocBuf(gWIn, (size_t)dm * is)) return false;
		if (!allocBuf(bIn, dm)) return false;
		if (!skipAdamBufs && !allocBuf(mBIn, dm)) return false;
		if (!skipAdamBufs && !allocBuf(v2BIn, dm)) return false;
		if (!allocBuf(gBIn, dm)) return false;
	}

	// Output projection
	if (!tokenModel)
	{
		if (!allocBuf(WOut, (size_t)os * dm)) return false;
		if (!skipAdamBufs && !allocBuf(vWOut, (size_t)os * dm)) return false;
		if (!skipAdamBufs && !allocBuf(v2WOut, (size_t)os * dm)) return false;
		if (!allocBuf(gWOut, (size_t)os * dm)) return false;
		if (!allocBuf(bOut, os)) return false;
		if (!skipAdamBufs && !allocBuf(mBOut, os)) return false;
		if (!skipAdamBufs && !allocBuf(v2BOut, os)) return false;
		if (!allocBuf(gBOut, os)) return false;
	}

	// Final LayerNorm
	if (!allocBuf(lnFinalGamma, dm)) return false;
	if (!allocBuf(lnFinalBeta, dm)) return false;
	if (!skipAdamBufs && !allocBuf(mLnFinalGamma, dm)) return false;
	if (!skipAdamBufs && !allocBuf(v2LnFinalGamma, dm)) return false;
	if (!skipAdamBufs && !allocBuf(mLnFinalBeta, dm)) return false;
	if (!skipAdamBufs && !allocBuf(v2LnFinalBeta, dm)) return false;
	if (!allocBuf(gLnFinalGamma, dm)) return false;
	if (!allocBuf(gLnFinalBeta, dm)) return false;

	// Blocks
	blocks = new Block[nl];
	for (unsigned int l = 0; l < nl; ++l)
	{
		Block& b = blocks[l];

		// LN1
		if (!allocBuf(b.ln1Gamma, dm)) return false;
		if (!allocBuf(b.ln1Beta, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.mLn1Gamma, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2Ln1Gamma, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.mLn1Beta, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2Ln1Beta, dm)) return false;
		if (!allocBuf(b.gLn1Gamma, dm)) return false;
		if (!allocBuf(b.gLn1Beta, dm)) return false;

		// QKV+O projections
		if (!allocBuf(b.Wq, (size_t)dm * dm)) return false;
		if (!allocBuf(b.Wk, (size_t)dm * dModelKV)) return false;
		if (!allocBuf(b.Wv, (size_t)dm * dModelKV)) return false;
		if (!allocBuf(b.Wo, (size_t)dm * dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.vWq, (size_t)dm * dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.vWk, (size_t)dm * dModelKV)) return false;
		if (!skipAdamBufs && !allocBuf(b.vWv, (size_t)dm * dModelKV)) return false;
		if (!skipAdamBufs && !allocBuf(b.vWo, (size_t)dm * dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2Wq, (size_t)dm * dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2Wk, (size_t)dm * dModelKV)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2Wv, (size_t)dm * dModelKV)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2Wo, (size_t)dm * dm)) return false;
		if (!allocBuf(b.gWq, (size_t)dm * dm)) return false;
		if (!allocBuf(b.gWk, (size_t)dm * dModelKV)) return false;
		if (!allocBuf(b.gWv, (size_t)dm * dModelKV)) return false;
		if (!allocBuf(b.gWo, (size_t)dm * dm)) return false;

		if (!allocBuf(b.bq, dm)) return false;
		if (!allocBuf(b.bk, dModelKV)) return false;
		if (!allocBuf(b.bv, dModelKV)) return false;
		if (!allocBuf(b.bo, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.mBq, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.mBk, dModelKV)) return false;
		if (!skipAdamBufs && !allocBuf(b.mBv, dModelKV)) return false;
		if (!skipAdamBufs && !allocBuf(b.mBo, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2Bq, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2Bk, dModelKV)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2Bv, dModelKV)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2Bo, dm)) return false;
		if (!allocBuf(b.gBq, dm)) return false;
		if (!allocBuf(b.gBk, dModelKV)) return false;
		if (!allocBuf(b.gBv, dModelKV)) return false;
		if (!allocBuf(b.gBo, dm)) return false;

		// LN2
		if (!allocBuf(b.ln2Gamma, dm)) return false;
		if (!allocBuf(b.ln2Beta, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.mLn2Gamma, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2Ln2Gamma, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.mLn2Beta, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2Ln2Beta, dm)) return false;
		if (!allocBuf(b.gLn2Gamma, dm)) return false;
		if (!allocBuf(b.gLn2Beta, dm)) return false;

		// FFN
		if (!allocBuf(b.W1, (size_t)ff1Width * dm)) return false;
		if (!allocBuf(b.W2, (size_t)dm * df)) return false;
		if (!skipAdamBufs && !allocBuf(b.vW1, (size_t)ff1Width * dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.vW2, (size_t)dm * df)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2W1, (size_t)ff1Width * dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2W2, (size_t)dm * df)) return false;
		if (!allocBuf(b.gW1, (size_t)ff1Width * dm)) return false;
		if (!allocBuf(b.gW2, (size_t)dm * df)) return false;
		if (!allocBuf(b.b1, ff1Width)) return false;
		if (!allocBuf(b.b2, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.mB1, ff1Width)) return false;
		if (!skipAdamBufs && !allocBuf(b.mB2, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2B1, ff1Width)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2B2, dm)) return false;
		if (!allocBuf(b.gB1, ff1Width)) return false;
		if (!allocBuf(b.gB2, dm)) return false;
	}

	// Allocate batched Adam device arrays (only needed for Adam optimizer).
	if (!skipAdamBufs)
	{
		int maxGroups = 6 + 16 * static_cast<int>(nl);
		cudaError_t e;
		e = cudaMalloc(&d_adamParams, maxGroups * sizeof(float*));  if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamGrads,  maxGroups * sizeof(float*));  if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamM,      maxGroups * sizeof(float*));  if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamV,      maxGroups * sizeof(float*));  if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamLr,     maxGroups * sizeof(float));   if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamWd,     maxGroups * sizeof(float));   if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamSizes,  maxGroups * sizeof(int));     if (e != cudaSuccess) return false;
		adamGroupCount = 0;
		adamMaxSize = 0;
		adamPtrsUploaded = false;
	}

	initialized = true;
	return true;
}

void GpuTransformerWeights::free()
{
	if (blocks)
	{
		delete[] blocks;
		blocks = 0;
	}
	if (d_adamParams) { cudaFree(d_adamParams); d_adamParams = 0; }
	if (d_adamGrads)  { cudaFree(d_adamGrads);  d_adamGrads  = 0; }
	if (d_adamM)      { cudaFree(d_adamM);      d_adamM      = 0; }
	if (d_adamV)      { cudaFree(d_adamV);       d_adamV      = 0; }
	if (d_adamLr)     { cudaFree(d_adamLr);      d_adamLr     = 0; }
	if (d_adamWd)     { cudaFree(d_adamWd);      d_adamWd     = 0; }
	if (d_adamSizes)  { cudaFree(d_adamSizes);   d_adamSizes  = 0; }
	adamGroupCount = 0;
	adamMaxSize = 0;
	adamPtrsUploaded = false;
	initialized = false;
	// GpuBuffer destructors handle cudaFree automatically.
}

// ---- GpuTransformerScratch ----

GpuTransformerScratch::GpuTransformerScratch()
    : initialized(false),
      T(0), dModel(0), dFF(0), dModelKV(0), nHeads(0), nLayers(0),
      inputSize(0), outSize(0), ff1Width(0),
      d_dKdVZeroPtrs(0), d_dKdVZeroSizes(0)
{
}

GpuTransformerScratch::~GpuTransformerScratch()
{
	free();
}

bool GpuTransformerScratch::allocate(unsigned int newT, unsigned int is, unsigned int os,
                                      unsigned int dm, unsigned int df, unsigned int dmkv,
                                      unsigned int nh, unsigned int nl, unsigned int f1w)
{
	free();

	T = newT;
	inputSize = is;
	outSize = os;
	dModel = dm;
	dFF = df;
	dModelKV = dmkv;
	nHeads = nh;
	nLayers = nl;
	ff1Width = f1w;

	const size_t sT = static_cast<size_t>(T);
	const size_t sdm = static_cast<size_t>(dm);
	const size_t sdf = static_cast<size_t>(df);
	const size_t sdmkv = static_cast<size_t>(dmkv);
	const size_t snl = static_cast<size_t>(nl);
	const size_t sf1w = static_cast<size_t>(f1w);
	const size_t sis = static_cast<size_t>(is);
	const size_t sos = static_cast<size_t>(os);

	// Forward
	if (!x.allocate(sT * sis)) return false;
	if (!h.allocate(sT * sdm)) return false;
	if (!ln1Mean.allocate(snl * sT)) return false;
	if (!ln1InvStd.allocate(snl * sT)) return false;
	if (!x1.allocate(snl * sT * sdm)) return false;
	if (!Q.allocate(snl * sT * sdm)) return false;
	if (!K.allocate(snl * sT * sdmkv)) return false;
	if (!V.allocate(snl * sT * sdmkv)) return false;
	if (!attnConcat.allocate(snl * sT * sdm)) return false;
	if (!attnOut.allocate(snl * sT * sdm)) return false;
	if (!hAfterAttn.allocate(snl * sT * sdm)) return false;
	if (!ln2Mean.allocate(snl * sT)) return false;
	if (!ln2InvStd.allocate(snl * sT)) return false;
	if (!x2.allocate(snl * sT * sdm)) return false;
	if (!ff1.allocate(snl * sT * sf1w)) return false;
	if (!ff1Act.allocate(snl * sT * sdf)) return false;
	if (!ffOut.allocate(snl * sT * sdm)) return false;
	if (!hAfterFF.allocate(snl * sT * sdm)) return false;
	if (!hPostFinalLN.allocate(sT * sdm)) return false;
	if (!lnFinalMean.allocate(sT)) return false;
	if (!lnFinalInvStd.allocate(sT)) return false;
	if (!logits.allocate(sT * sos)) return false;
	if (!probs.allocate(sT * sos)) return false;

	// Backward
	if (!dLogits.allocate(sT * sos)) return false;
	if (!dH.allocate(sT * sdm)) return false;
	if (!dH2.allocate(sT * sdm)) return false;
	if (!dFF1Act.allocate(sT * sdf)) return false;
	if (!dFF1Cat.allocate(sT * sf1w)) return false;
	if (!dX2.allocate(sT * sdm)) return false;
	if (!dHAfterAttnFromLN.allocate(sT * sdm)) return false;
	if (!dAttnConcat.allocate(sT * sdm)) return false;
	if (!dQfull.allocate(sT * sdm)) return false;
	if (!dKfull.allocate(sT * sdmkv)) return false;
	if (!dVfull.allocate(sT * sdmkv)) return false;
	if (!dX1.allocate(sT * sdm)) return false;
	if (!dXtmp.allocate(sT * sdm)) return false;
	if (!dHInFromLN.allocate(sT * sdm)) return false;
	if (!dInput.allocate(sT * sis)) return false;

	// Token IDs
	if (!tokenIds.allocate(sT)) return false;

	// Persistent per-step buffers
	const size_t snh = static_cast<size_t>(nh);
	const size_t dHead = sdm / snh;
	if (!gpuInvFreq.allocate(dHead / 2)) return false;
	if (!gpuTargetsT.allocate(sT)) return false;

	// GPU loss computation scalars
	if (!lossSum.allocate(1)) return false;
	if (!lossCount.allocate(1)) return false;
	if (!correctCount.allocate(1)) return false;
	if (!validCount.allocate(1)) return false;
	if (!lossPack.allocate(4)) return false;

	// Persistent device arrays for batch-zeroing dK/dV.
	{
		cudaError_t e1 = cudaMalloc(&d_dKdVZeroPtrs, 2 * sizeof(float*));
		cudaError_t e2 = cudaMalloc(&d_dKdVZeroSizes, 2 * sizeof(int));
		if (e1 != cudaSuccess || e2 != cudaSuccess) return false;
		float* hPtrs[2] = { dKfull.data(), dVfull.data() };
		int hSizes[2] = { static_cast<int>(sT * sdmkv), static_cast<int>(sT * sdmkv) };
		cudaMemcpy(d_dKdVZeroPtrs, hPtrs, 2 * sizeof(float*), cudaMemcpyHostToDevice);
		cudaMemcpy(d_dKdVZeroSizes, hSizes, 2 * sizeof(int), cudaMemcpyHostToDevice);
	}

	initialized = true;
	return true;
}

void GpuTransformerScratch::free()
{
	initialized = false;
	if (d_dKdVZeroPtrs)  { cudaFree(d_dKdVZeroPtrs);  d_dKdVZeroPtrs  = 0; }
	if (d_dKdVZeroSizes) { cudaFree(d_dKdVZeroSizes); d_dKdVZeroSizes = 0; }
	// All GpuBuffer destructors handle cudaFree.
}

// ---- Upload/download helpers ----

bool uploadTransformerWeights(GpuTransformerWeights& gpu,
                               const float* tokE, size_t tokESize,
                               const float* WIn, size_t WInSize,
                               const float* bIn, size_t bInSize,
                               const float* WOut, size_t WOutSize,
                               const float* bOut, size_t bOutSize,
                               const float* lmBias, size_t lmBiasSize,
                               const float* lnFinalGamma, size_t lnFinalGammaSize,
                               const float* lnFinalBeta, size_t lnFinalBetaSize)
{
	if (!gpu.initialized)
		return false;

	if (gpu.tokenModel && tokE && tokESize > 0)
	{
		if (!gpu.tokE.upload(tokE, tokESize)) return false;
	}
	if (!gpu.tokenModel && WIn && WInSize > 0)
	{
		if (!gpu.WIn.upload(WIn, WInSize)) return false;
	}
	if (!gpu.tokenModel && bIn && bInSize > 0)
	{
		if (!gpu.bIn.upload(bIn, bInSize)) return false;
	}
	if (!gpu.tokenModel && WOut && WOutSize > 0)
	{
		if (!gpu.WOut.upload(WOut, WOutSize)) return false;
	}
	if (!gpu.tokenModel && bOut && bOutSize > 0)
	{
		if (!gpu.bOut.upload(bOut, bOutSize)) return false;
	}
	if (gpu.tokenModel && lmBias && lmBiasSize > 0)
	{
		if (!gpu.lmBias.upload(lmBias, lmBiasSize)) return false;
	}
	if (lnFinalGamma && lnFinalGammaSize > 0)
	{
		if (!gpu.lnFinalGamma.upload(lnFinalGamma, lnFinalGammaSize)) return false;
	}
	if (lnFinalBeta && lnFinalBetaSize > 0)
	{
		if (!gpu.lnFinalBeta.upload(lnFinalBeta, lnFinalBetaSize)) return false;
	}

	return true;
}

bool downloadTransformerWeights(const GpuTransformerWeights& gpu,
                                 float* tokE, size_t tokESize,
                                 float* WIn, size_t WInSize,
                                 float* bIn, size_t bInSize,
                                 float* WOut, size_t WOutSize,
                                 float* bOut, size_t bOutSize,
                                 float* lmBias, size_t lmBiasSize,
                                 float* lnFinalGamma, size_t lnFinalGammaSize,
                                 float* lnFinalBeta, size_t lnFinalBetaSize)
{
	if (!gpu.initialized)
		return false;

	if (gpu.tokenModel && tokE && tokESize > 0)
	{
		if (!gpu.tokE.download(tokE, tokESize)) return false;
	}
	if (!gpu.tokenModel && WIn && WInSize > 0)
	{
		if (!gpu.WIn.download(WIn, WInSize)) return false;
	}
	if (!gpu.tokenModel && bIn && bInSize > 0)
	{
		if (!gpu.bIn.download(bIn, bInSize)) return false;
	}
	if (!gpu.tokenModel && WOut && WOutSize > 0)
	{
		if (!gpu.WOut.download(WOut, WOutSize)) return false;
	}
	if (!gpu.tokenModel && bOut && bOutSize > 0)
	{
		if (!gpu.bOut.download(bOut, bOutSize)) return false;
	}
	if (gpu.tokenModel && lmBias && lmBiasSize > 0)
	{
		if (!gpu.lmBias.download(lmBias, lmBiasSize)) return false;
	}
	if (lnFinalGamma && lnFinalGammaSize > 0)
	{
		if (!gpu.lnFinalGamma.download(lnFinalGamma, lnFinalGammaSize)) return false;
	}
	if (lnFinalBeta && lnFinalBetaSize > 0)
	{
		if (!gpu.lnFinalBeta.download(lnFinalBeta, lnFinalBetaSize)) return false;
	}

	return true;
}

bool uploadTransformerBlockWeights(GpuTransformerWeights::Block& b,
                                    unsigned int dModel, unsigned int dModelKV,
                                    unsigned int ff1Width, unsigned int dFF,
                                    const float* ln1Gamma, const float* ln1Beta,
                                    const float* Wq, const float* Wk, const float* Wv, const float* Wo,
                                    const float* bq, const float* bk, const float* bv, const float* bo,
                                    const float* ln2Gamma, const float* ln2Beta,
                                    const float* W1, const float* W2,
                                    const float* b1, const float* b2)
{
	const size_t dm = static_cast<size_t>(dModel);
	const size_t dmkv = static_cast<size_t>(dModelKV);
	const size_t f1w = static_cast<size_t>(ff1Width);
	const size_t df = static_cast<size_t>(dFF);

	if (!b.ln1Gamma.upload(ln1Gamma, dm)) return false;
	if (!b.ln1Beta.upload(ln1Beta, dm)) return false;
	if (!b.Wq.upload(Wq, dm * dm)) return false;
	if (!b.Wk.upload(Wk, dm * dmkv)) return false;
	if (!b.Wv.upload(Wv, dm * dmkv)) return false;
	if (!b.Wo.upload(Wo, dm * dm)) return false;
	if (!b.bq.upload(bq, dm)) return false;
	if (!b.bk.upload(bk, dmkv)) return false;
	if (!b.bv.upload(bv, dmkv)) return false;
	if (!b.bo.upload(bo, dm)) return false;
	if (!b.ln2Gamma.upload(ln2Gamma, dm)) return false;
	if (!b.ln2Beta.upload(ln2Beta, dm)) return false;
	if (!b.W1.upload(W1, f1w * dm)) return false;
	if (!b.W2.upload(W2, dm * df)) return false;
	if (!b.b1.upload(b1, f1w)) return false;
	if (!b.b2.upload(b2, dm)) return false;

	return true;
}

// Helper: append a GpuBuffer to the batch-zero list if allocated.
static void addBuf(GpuBuffer<float>& buf, float** hPtrs, int* hSizes, int& count)
{
	if (buf.allocated())
	{
		hPtrs[count] = buf.data();
		hSizes[count] = static_cast<int>(buf.size());
		++count;
	}
}

bool zeroTransformerGradients(GpuTransformerWeights& gpu)
{
	if (!gpu.initialized)
		return false;

	// Max buffers: 6 global + 16 per layer (256 layers max).
	float* hPtrs[6 + 16 * 256];
	int    hSizes[6 + 16 * 256];
	int count = 0;

	if (gpu.tokenModel)
	{
		addBuf(gpu.gTokE, hPtrs, hSizes, count);
		addBuf(gpu.gLmBias, hPtrs, hSizes, count);
	}
	else
	{
		addBuf(gpu.gWIn, hPtrs, hSizes, count);
		addBuf(gpu.gBIn, hPtrs, hSizes, count);
		addBuf(gpu.gWOut, hPtrs, hSizes, count);
		addBuf(gpu.gBOut, hPtrs, hSizes, count);
	}

	for (unsigned int l = 0; l < gpu.nLayers; ++l)
	{
		GpuTransformerWeights::Block& b = gpu.blocks[l];
		addBuf(b.gLn1Gamma, hPtrs, hSizes, count);
		addBuf(b.gLn1Beta, hPtrs, hSizes, count);
		addBuf(b.gWq, hPtrs, hSizes, count);
		addBuf(b.gWk, hPtrs, hSizes, count);
		addBuf(b.gWv, hPtrs, hSizes, count);
		addBuf(b.gWo, hPtrs, hSizes, count);
		addBuf(b.gBq, hPtrs, hSizes, count);
		addBuf(b.gBk, hPtrs, hSizes, count);
		addBuf(b.gBv, hPtrs, hSizes, count);
		addBuf(b.gBo, hPtrs, hSizes, count);
		addBuf(b.gLn2Gamma, hPtrs, hSizes, count);
		addBuf(b.gLn2Beta, hPtrs, hSizes, count);
		addBuf(b.gW1, hPtrs, hSizes, count);
		addBuf(b.gW2, hPtrs, hSizes, count);
		addBuf(b.gB1, hPtrs, hSizes, count);
		addBuf(b.gB2, hPtrs, hSizes, count);
	}

	addBuf(gpu.gLnFinalGamma, hPtrs, hSizes, count);
	addBuf(gpu.gLnFinalBeta, hPtrs, hSizes, count);

	if (count == 0)
		return true;

	// Persistent device buffers for the pointer/size arrays (re-allocated if needed).
	static float** d_ptrs = 0;
	static int*    d_sizes = 0;
	static int     d_capacity = 0;

	if (count > d_capacity)
	{
		if (d_ptrs)  cudaFree(d_ptrs);
		if (d_sizes) cudaFree(d_sizes);
		cudaMalloc(&d_ptrs,  count * sizeof(float*));
		cudaMalloc(&d_sizes, count * sizeof(int));
		d_capacity = count;
	}

	cudaMemcpy(d_ptrs,  hPtrs,  count * sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_sizes, hSizes, count * sizeof(int),    cudaMemcpyHostToDevice);

	return zero_buffers_batch(d_ptrs, d_sizes, count);
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
