// GPU transformer state implementation.
#include "gpu_transformer_state.h"
#include "gpu_kernels.h"
#include <cmath>

#ifdef GLADES_HAVE_CUDA

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
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
      d_adamRowSecond(0), d_adamColSecond(0),
      d_adamRowMetric(0), d_adamColMetric(0),
      d_adamRowStructMetric(0), d_adamColStructMetric(0),
      d_adamPrevMhat(0), d_adamMetricScratch(0), d_echoObserveEntries(0),
      d_matraBatchItems(0), d_matraStatsBatch(0),
      d_matraCoreBatchPtrs(0), d_matraStepBatchPtrs(0), d_matraInfoBatch(0),
      d_muonBatchItems(0),
      d_muonCoreBatchPtrs(0), d_muonStepBatchPtrs(0), d_muonInfoBatch(0),
      d_adamBaseLr(0), d_adamWd(0), d_adamGroupScales(0), d_adamGroupPrevStepRms(0), d_adamSizes(0),
      d_adamMetricRows(0), d_adamMetricCols(0),
      adamGroupCount(0), adamMaxSize(0), echoObserveCapacity(0), matraCoreBatchCapacity(0), muonCoreBatchCapacity(0),
      echoObserveEntryCount(0), echoObserveTotalFeatures(0),
      echoObserveSeqLen(0u), echoObserveScope(0u), echoObserveTokenModel(false),
      echoObserveMetaUploaded(false),
      matraBatchDescriptorsUploaded(false),
      adamPtrsUploaded(false),
      adamMetricMetaUploaded(false), adamMetricScope(0u),
      matraBatchDescriptorCount(0), matraBatchDescriptorHash(0ULL),
      lowpReady(false), lowpDType(0), lowpIsCanonical(false)
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

static bool allocBuf(GpuBuffer<uint16_t>& buf, size_t n)
{
	if (n == 0)
		return true;
	return buf.allocate(n);
}

static bool allocBuf(GpuBuffer<int8_t>& buf, size_t n)
{
	if (n == 0)
		return true;
	return buf.allocate(n);
}

static bool allocBuf(GpuBuffer<uint8_t>& buf, size_t n)
{
	if (n == 0)
		return true;
	return buf.allocate(n);
}

bool GpuTransformerWeights::allocate(unsigned int dm, unsigned int df, unsigned int nh,
                                      unsigned int nkvh, unsigned int nl,
                                      unsigned int vs, unsigned int is, unsigned int os,
                                      unsigned int ffk, bool tm, bool te,
                                      bool skipAdamBufs,
                                      bool adamStateBf16,
                                      int mlaLatentDim,
                                      bool adamStateInt8,
                                      bool faceEmbedding,
                                      bool gradStorageBf16,
                                      bool gradStorageBf16Phase2,
                                      bool weightStorageBf16)
{
	free();
	// int8 wins over bf16 if both flags accidentally set (it's the more
	// aggressive compression — see MixedPrecisionConfig comments).
	const bool useInt8    = adamStateInt8 && !skipAdamBufs;
	const bool useBf16    = adamStateBf16 && !skipAdamBufs && !useInt8;
	// FACE on tokE replaces ALL dense Adam state on the embedding (m, v
	// in any precision).  When tm && faceEmbedding, allocate FACE state
	// instead of vTokE/v2TokE/vTokE_bf16/etc.
	const bool useFaceTokE = faceEmbedding && tm && !skipAdamBufs;
	const bool useBf16Grads = gradStorageBf16 && !skipAdamBufs;
	const bool useBf16GradsPh2 = gradStorageBf16Phase2 && useBf16Grads;
	const bool useBf16Weights_ = weightStorageBf16 && useBf16GradsPh2;
	// Set the canonical-bf16 flag here so ensureLowpMirrors short-circuits
	// (mirrors are the canonical store, no FP32 master refresh).
	lowpIsCanonical = useBf16Weights_;
	// Phase-2: per-block W{q,k,v,o,1,2}, gWIn, gWOut FP32 grad buffers are
	// RETIRED (backward writes scratch+commit-bf16 directly).  Bias grads
	// and gTokE keep their FP32 allocs (gTokE goes through Phase-1 cast
	// path due to the bf16-scatter precision issue).  All retired by default
	// in Phase-2; the GLADES_BF16_PH2_RETIRE env var (off|w2|w1|wq|wk|wv|wo|win|wout|all)
	// remains as a per-tensor diagnostic override for debugging.
	const char* phase2Mode_env = useBf16GradsPh2 ? std::getenv("GLADES_BF16_PH2_RETIRE") : NULL;
	const bool ph2DiagOff   = phase2Mode_env && !std::strcmp(phase2Mode_env, "off");
	const bool ph2RetireAll = useBf16GradsPh2 && !ph2DiagOff
	    && (phase2Mode_env == NULL || !std::strcmp(phase2Mode_env, "all"));
	const bool ph2RetireW2  = ph2RetireAll || (phase2Mode_env && !std::strcmp(phase2Mode_env, "w2"));
	const bool ph2RetireW1  = ph2RetireAll || (phase2Mode_env && !std::strcmp(phase2Mode_env, "w1"));
	const bool ph2RetireWq  = ph2RetireAll || (phase2Mode_env && !std::strcmp(phase2Mode_env, "wq"));
	const bool ph2RetireWk  = ph2RetireAll || (phase2Mode_env && !std::strcmp(phase2Mode_env, "wk"));
	const bool ph2RetireWv  = ph2RetireAll || (phase2Mode_env && !std::strcmp(phase2Mode_env, "wv"));
	const bool ph2RetireWo  = ph2RetireAll || (phase2Mode_env && !std::strcmp(phase2Mode_env, "wo"));
	const bool ph2RetireWIn = ph2RetireAll || (phase2Mode_env && !std::strcmp(phase2Mode_env, "win"));
	const bool ph2RetireWOut= ph2RetireAll || (phase2Mode_env && !std::strcmp(phase2Mode_env, "wout"));
	const bool allocFpMV  = !skipAdamBufs && !useBf16 && !useInt8;
	const bool allocBfMV  = useBf16;
	const bool allocI8MV  = useInt8;
	// Per-256-element absmax-scale array length (matches ADAM_INT8_BS in the
	// kernel).  Kept inline here so this header doesn't need to reach into
	// gpu_kernels.h for the constant.
	#define I8_SCALE_N(n_) (((n_) + 255UL) >> 8)

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
		// FACE replaces all dense Adam state on tokE — skip the m/v allocations
		// in that case and instead allocate FACE state.  The dispatch in
		// sgd_transformer.cpp picks FACE over int8/bf16/FP32 when faceEmbedding
		// is set.
		const bool tokE_fp_mv = allocFpMV && !useFaceTokE;
		const bool tokE_bf_mv = allocBfMV && !useFaceTokE;
		const bool tokE_i8_mv = allocI8MV && !useFaceTokE;
		if (tokE_fp_mv && !allocBuf(vTokE, (size_t)vs * dm)) return false;
		if (tokE_fp_mv && !allocBuf(v2TokE, (size_t)vs * dm)) return false;
		if (tokE_bf_mv && !allocBuf(vTokE_bf16, (size_t)vs * dm)) return false;
		if (tokE_bf_mv && !allocBuf(v2TokE_bf16, (size_t)vs * dm)) return false;
		if (tokE_i8_mv) {
			const size_t n_ = (size_t)vs * dm;
			if (!allocBuf(vTokE_int8,   n_))               return false;
			if (!allocBuf(v2TokE_int8,  n_))               return false;
			if (!allocBuf(vTokEScale,   I8_SCALE_N(n_)))   return false;
			if (!allocBuf(v2TokEScale,  I8_SCALE_N(n_)))   return false;
		}
		if (useFaceTokE) {
			if (!allocBuf(faceZnBar,  (size_t)vs))  return false;
			if (!allocBuf(faceDnBar,  (size_t)dm))  return false;
			if (!allocBuf(faceQHat,   (size_t)1))   return false;
			if (!allocBuf(faceGFHat,  (size_t)1))   return false;
			if (!allocBuf(faceZnNew,  (size_t)vs))  return false;
			if (!allocBuf(faceDnRaw,  (size_t)dm))  return false;
			if (!allocBuf(faceQStep,  (size_t)1))   return false;
			if (!allocBuf(faceGFStep, (size_t)1))   return false;
		}
		if (!allocBuf(gTokE, (size_t)vs * dm)) return false;
		if (useBf16Grads && !allocBuf(gTokE_bf16, (size_t)vs * dm)) return false;
		if (!allocBuf(lmBias, vs)) return false;
		if (!skipAdamBufs && !allocBuf(mLmBias, vs)) return false;
		if (!skipAdamBufs && !allocBuf(v2LmBias, vs)) return false;
		if (!allocBuf(gLmBias, vs)) return false;
	}

	// Input projection
	if (!tokenModel)
	{
		if (!allocBuf(WIn, (size_t)dm * is)) return false;
		if (allocFpMV && !allocBuf(vWIn, (size_t)dm * is)) return false;
		if (allocFpMV && !allocBuf(v2WIn, (size_t)dm * is)) return false;
		if (allocBfMV && !allocBuf(vWIn_bf16, (size_t)dm * is)) return false;
		if (allocBfMV && !allocBuf(v2WIn_bf16, (size_t)dm * is)) return false;
		if (allocI8MV) {
			const size_t n_ = (size_t)dm * is;
			if (!allocBuf(vWIn_int8,   n_))               return false;
			if (!allocBuf(v2WIn_int8,  n_))               return false;
			if (!allocBuf(vWInScale,   I8_SCALE_N(n_)))   return false;
			if (!allocBuf(v2WInScale,  I8_SCALE_N(n_)))   return false;
		}
		if (!ph2RetireWIn && !allocBuf(gWIn, (size_t)dm * is)) return false;
		if (useBf16Grads && !allocBuf(gWIn_bf16, (size_t)dm * is)) return false;
		if (!allocBuf(bIn, dm)) return false;
		if (!skipAdamBufs && !allocBuf(mBIn, dm)) return false;
		if (!skipAdamBufs && !allocBuf(v2BIn, dm)) return false;
		if (!allocBuf(gBIn, dm)) return false;
	}

	// Output projection
	if (!tokenModel)
	{
		if (!allocBuf(WOut, (size_t)os * dm)) return false;
		if (allocFpMV && !allocBuf(vWOut, (size_t)os * dm)) return false;
		if (allocFpMV && !allocBuf(v2WOut, (size_t)os * dm)) return false;
		if (allocBfMV && !allocBuf(vWOut_bf16, (size_t)os * dm)) return false;
		if (allocBfMV && !allocBuf(v2WOut_bf16, (size_t)os * dm)) return false;
		if (allocI8MV) {
			const size_t n_ = (size_t)os * dm;
			if (!allocBuf(vWOut_int8,   n_))               return false;
			if (!allocBuf(v2WOut_int8,  n_))               return false;
			if (!allocBuf(vWOutScale,   I8_SCALE_N(n_)))   return false;
			if (!allocBuf(v2WOutScale,  I8_SCALE_N(n_)))   return false;
		}
		if (!ph2RetireWOut && !allocBuf(gWOut, (size_t)os * dm)) return false;
		if (useBf16Grads && !allocBuf(gWOut_bf16, (size_t)os * dm)) return false;
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
		if (allocFpMV && !allocBuf(b.vWq, (size_t)dm * dm)) return false;
		if (allocFpMV && !allocBuf(b.vWk, (size_t)dm * dModelKV)) return false;
		if (allocFpMV && !allocBuf(b.vWv, (size_t)dm * dModelKV)) return false;
		if (allocFpMV && !allocBuf(b.vWo, (size_t)dm * dm)) return false;
		if (allocFpMV && !allocBuf(b.v2Wq, (size_t)dm * dm)) return false;
		if (allocFpMV && !allocBuf(b.v2Wk, (size_t)dm * dModelKV)) return false;
		if (allocFpMV && !allocBuf(b.v2Wv, (size_t)dm * dModelKV)) return false;
		if (allocFpMV && !allocBuf(b.v2Wo, (size_t)dm * dm)) return false;
		if (allocBfMV && !allocBuf(b.vWq_bf16, (size_t)dm * dm)) return false;
		if (allocBfMV && !allocBuf(b.vWk_bf16, (size_t)dm * dModelKV)) return false;
		if (allocBfMV && !allocBuf(b.vWv_bf16, (size_t)dm * dModelKV)) return false;
		if (allocBfMV && !allocBuf(b.vWo_bf16, (size_t)dm * dm)) return false;
		if (allocBfMV && !allocBuf(b.v2Wq_bf16, (size_t)dm * dm)) return false;
		if (allocBfMV && !allocBuf(b.v2Wk_bf16, (size_t)dm * dModelKV)) return false;
		if (allocBfMV && !allocBuf(b.v2Wv_bf16, (size_t)dm * dModelKV)) return false;
		if (allocBfMV && !allocBuf(b.v2Wo_bf16, (size_t)dm * dm)) return false;
		if (allocI8MV) {
			const size_t nQ = (size_t)dm * dm;
			const size_t nK = (size_t)dm * dModelKV;
			const size_t nV = (size_t)dm * dModelKV;
			const size_t nO = (size_t)dm * dm;
			if (!allocBuf(b.vWq_int8,   nQ))             return false;
			if (!allocBuf(b.vWk_int8,   nK))             return false;
			if (!allocBuf(b.vWv_int8,   nV))             return false;
			if (!allocBuf(b.vWo_int8,   nO))             return false;
			if (!allocBuf(b.v2Wq_int8,  nQ))             return false;
			if (!allocBuf(b.v2Wk_int8,  nK))             return false;
			if (!allocBuf(b.v2Wv_int8,  nV))             return false;
			if (!allocBuf(b.v2Wo_int8,  nO))             return false;
			if (!allocBuf(b.vWqScale,   I8_SCALE_N(nQ))) return false;
			if (!allocBuf(b.vWkScale,   I8_SCALE_N(nK))) return false;
			if (!allocBuf(b.vWvScale,   I8_SCALE_N(nV))) return false;
			if (!allocBuf(b.vWoScale,   I8_SCALE_N(nO))) return false;
			if (!allocBuf(b.v2WqScale,  I8_SCALE_N(nQ))) return false;
			if (!allocBuf(b.v2WkScale,  I8_SCALE_N(nK))) return false;
			if (!allocBuf(b.v2WvScale,  I8_SCALE_N(nV))) return false;
			if (!allocBuf(b.v2WoScale,  I8_SCALE_N(nO))) return false;
		}
		if (!ph2RetireWq && !allocBuf(b.gWq, (size_t)dm * dm)) return false;
		if (!ph2RetireWk && !allocBuf(b.gWk, (size_t)dm * dModelKV)) return false;
		if (!ph2RetireWv && !allocBuf(b.gWv, (size_t)dm * dModelKV)) return false;
		if (!ph2RetireWo && !allocBuf(b.gWo, (size_t)dm * dm)) return false;
		if (useBf16Grads) {
			if (!allocBuf(b.gWq_bf16, (size_t)dm * dm)) return false;
			if (!allocBuf(b.gWk_bf16, (size_t)dm * dModelKV)) return false;
			if (!allocBuf(b.gWv_bf16, (size_t)dm * dModelKV)) return false;
			if (!allocBuf(b.gWo_bf16, (size_t)dm * dm)) return false;
		}

		// Paradigm shift #76 MLA latent projections (allocated when mlaLatentDim > 0).
		if (mlaLatentDim > 0) {
			const size_t dC = (size_t)mlaLatentDim;
			if (!allocBuf(b.Wdkv, (size_t)dm * dC)) return false;
			if (!allocBuf(b.Wuk,  dC * (size_t)dModelKV)) return false;
			if (!allocBuf(b.Wuv,  dC * (size_t)dModelKV)) return false;
			if (allocFpMV) {
				if (!allocBuf(b.vWdkv,  (size_t)dm * dC)) return false;
				if (!allocBuf(b.vWuk,   dC * (size_t)dModelKV)) return false;
				if (!allocBuf(b.vWuv,   dC * (size_t)dModelKV)) return false;
				if (!allocBuf(b.v2Wdkv, (size_t)dm * dC)) return false;
				if (!allocBuf(b.v2Wuk,  dC * (size_t)dModelKV)) return false;
				if (!allocBuf(b.v2Wuv,  dC * (size_t)dModelKV)) return false;
			}
			if (!allocBuf(b.gWdkv, (size_t)dm * dC)) return false;
			if (!allocBuf(b.gWuk,  dC * (size_t)dModelKV)) return false;
			if (!allocBuf(b.gWuv,  dC * (size_t)dModelKV)) return false;
			if (useBf16Grads) {
				if (!allocBuf(b.gWdkv_bf16, (size_t)dm * dC)) return false;
				if (!allocBuf(b.gWuk_bf16,  dC * (size_t)dModelKV)) return false;
				if (!allocBuf(b.gWuv_bf16,  dC * (size_t)dModelKV)) return false;
			}
			// Forward scratch sized per batch — actual size T*dC depends on
			// runtime T; allocate at upper bound T_max via gpuTransformerScratch
			// instead. Mark these as zero-allocated here.
		}

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
		if (allocFpMV && !allocBuf(b.vW1, (size_t)ff1Width * dm)) return false;
		if (allocFpMV && !allocBuf(b.vW2, (size_t)dm * df)) return false;
		if (allocFpMV && !allocBuf(b.v2W1, (size_t)ff1Width * dm)) return false;
		if (allocFpMV && !allocBuf(b.v2W2, (size_t)dm * df)) return false;
		if (allocBfMV && !allocBuf(b.vW1_bf16, (size_t)ff1Width * dm)) return false;
		if (allocBfMV && !allocBuf(b.vW2_bf16, (size_t)dm * df)) return false;
		if (allocBfMV && !allocBuf(b.v2W1_bf16, (size_t)ff1Width * dm)) return false;
		if (allocBfMV && !allocBuf(b.v2W2_bf16, (size_t)dm * df)) return false;
		if (allocI8MV) {
			const size_t n1 = (size_t)ff1Width * dm;
			const size_t n2 = (size_t)dm * df;
			if (!allocBuf(b.vW1_int8,   n1))             return false;
			if (!allocBuf(b.vW2_int8,   n2))             return false;
			if (!allocBuf(b.v2W1_int8,  n1))             return false;
			if (!allocBuf(b.v2W2_int8,  n2))             return false;
			if (!allocBuf(b.vW1Scale,   I8_SCALE_N(n1))) return false;
			if (!allocBuf(b.vW2Scale,   I8_SCALE_N(n2))) return false;
			if (!allocBuf(b.v2W1Scale,  I8_SCALE_N(n1))) return false;
			if (!allocBuf(b.v2W2Scale,  I8_SCALE_N(n2))) return false;
		}
		if (!ph2RetireW1 && !allocBuf(b.gW1, (size_t)ff1Width * dm)) return false;
		if (!ph2RetireW2 && !allocBuf(b.gW2, (size_t)dm * df)) return false;
		if (useBf16Grads) {
			if (!allocBuf(b.gW1_bf16, (size_t)ff1Width * dm)) return false;
			if (!allocBuf(b.gW2_bf16, (size_t)dm * df)) return false;
		}
		if (!allocBuf(b.b1, ff1Width)) return false;
		if (!allocBuf(b.b2, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.mB1, ff1Width)) return false;
		if (!skipAdamBufs && !allocBuf(b.mB2, dm)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2B1, ff1Width)) return false;
		if (!skipAdamBufs && !allocBuf(b.v2B2, dm)) return false;
		if (!allocBuf(b.gB1, ff1Width)) return false;
		if (!allocBuf(b.gB2, dm)) return false;
	}

	// Allocate batched Adam device arrays shared by AdamW-like backbones.
	// ECHO-specific batched observe / metric metadata is allocated lazily when
	// the fused ECHO path is actually active.
	// Match sgd_transformer.cpp's maxAdamGroups: base 6 + 16 per layer + 3
	// per layer when MLA is active. Without MLA, the +3 is unused but
	// over-allocating a few pointer slots is harmless.
	const int adamPerLayer = 16 + (mlaLatentDim > 0 ? 3 : 0);
	int maxGroups = 6 + adamPerLayer * static_cast<int>(nl);
	cudaError_t e;
	echoObserveCapacity = 0;
	echoObserveEntryCount = 0;
	echoObserveTotalFeatures = 0;
	echoObserveSeqLen = 0u;
	echoObserveScope = 0u;
	echoObserveTokenModel = false;
	echoObserveMetaUploaded = false;

	if (!skipAdamBufs)
	{
		e = cudaMalloc(&d_adamParams, maxGroups * sizeof(float*));  if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamGrads,  maxGroups * sizeof(float*));  if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamM,      maxGroups * sizeof(float*));  if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamV,      maxGroups * sizeof(float*));  if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamBaseLr, maxGroups * sizeof(float));   if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamWd,     maxGroups * sizeof(float));   if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamGroupScales, maxGroups * sizeof(float)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamGroupPrevStepRms, maxGroups * sizeof(float)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamSizes,  maxGroups * sizeof(int));     if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_matraBatchItems, maxGroups * sizeof(GpuMatraBatchItem)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_matraStatsBatch, static_cast<size_t>(maxGroups) * 20u * sizeof(float)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_matraCoreBatchPtrs, maxGroups * sizeof(float*)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_matraStepBatchPtrs, maxGroups * sizeof(float*)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_matraInfoBatch, maxGroups * sizeof(int)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_muonBatchItems, maxGroups * sizeof(GpuMuonBatchItem)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_muonCoreBatchPtrs, maxGroups * sizeof(float*)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_muonStepBatchPtrs, maxGroups * sizeof(float*)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_muonInfoBatch, maxGroups * sizeof(int)); if (e != cudaSuccess) return false;
		cudaMemset(d_adamGroupScales, 0, maxGroups * sizeof(float));
		cudaMemset(d_adamGroupPrevStepRms, 0, maxGroups * sizeof(float));
		cudaMemset(d_matraBatchItems, 0, maxGroups * sizeof(GpuMatraBatchItem));
		cudaMemset(d_matraStatsBatch, 0, static_cast<size_t>(maxGroups) * 20u * sizeof(float));
		cudaMemset(d_matraCoreBatchPtrs, 0, maxGroups * sizeof(float*));
		cudaMemset(d_matraStepBatchPtrs, 0, maxGroups * sizeof(float*));
		cudaMemset(d_matraInfoBatch, 0, maxGroups * sizeof(int));
		cudaMemset(d_muonBatchItems, 0, maxGroups * sizeof(GpuMuonBatchItem));
		cudaMemset(d_muonCoreBatchPtrs, 0, maxGroups * sizeof(float*));
		cudaMemset(d_muonStepBatchPtrs, 0, maxGroups * sizeof(float*));
		cudaMemset(d_muonInfoBatch, 0, maxGroups * sizeof(int));
		adamGroupCount = 0;
		adamMaxSize = 0;
		matraCoreBatchCapacity = maxGroups;
		muonCoreBatchCapacity = maxGroups;
		matraBatchDescriptorsUploaded = false;
		matraBatchDescriptorCount = 0;
		matraBatchDescriptorHash = 0ULL;
		adamPtrsUploaded = false;
		adamMetricMetaUploaded = false;
		adamMetricScope = 0u;
	}

	initialized = true;
	return true;
}

bool GpuTransformerWeights::ensureEchoBuffers()
{
	const int maxGroups = 6 + 16 * static_cast<int>(nLayers);
	cudaError_t e = cudaSuccess;

	if (!d_echoObserveEntries)
	{
		e = cudaMalloc(&d_echoObserveEntries, static_cast<size_t>(maxGroups) * sizeof(GpuEchoObserveEntry));
		if (e != cudaSuccess)
			return false;
		echoObserveCapacity = maxGroups;
		echoObserveEntryCount = 0;
		echoObserveTotalFeatures = 0;
		echoObserveSeqLen = 0u;
		echoObserveScope = 0u;
		echoObserveTokenModel = false;
		echoObserveMetaUploaded = false;
	}

	if (!d_adamRowSecond)
	{
		e = cudaMalloc(&d_adamRowSecond, maxGroups * sizeof(float*)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamColSecond, maxGroups * sizeof(float*)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamRowMetric, maxGroups * sizeof(float*)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamColMetric, maxGroups * sizeof(float*)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamRowStructMetric, maxGroups * sizeof(float*)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamColStructMetric, maxGroups * sizeof(float*)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamPrevMhat, maxGroups * sizeof(float*)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamMetricScratch, maxGroups * sizeof(float*)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamMetricRows, maxGroups * sizeof(int)); if (e != cudaSuccess) return false;
		e = cudaMalloc(&d_adamMetricCols, maxGroups * sizeof(int)); if (e != cudaSuccess) return false;
		cudaMemset(d_adamRowSecond, 0, maxGroups * sizeof(float*));
		cudaMemset(d_adamColSecond, 0, maxGroups * sizeof(float*));
		cudaMemset(d_adamRowMetric, 0, maxGroups * sizeof(float*));
		cudaMemset(d_adamColMetric, 0, maxGroups * sizeof(float*));
		cudaMemset(d_adamRowStructMetric, 0, maxGroups * sizeof(float*));
		cudaMemset(d_adamColStructMetric, 0, maxGroups * sizeof(float*));
		cudaMemset(d_adamPrevMhat, 0, maxGroups * sizeof(float*));
		cudaMemset(d_adamMetricScratch, 0, maxGroups * sizeof(float*));
		cudaMemset(d_adamMetricRows, 0, maxGroups * sizeof(int));
		cudaMemset(d_adamMetricCols, 0, maxGroups * sizeof(int));
		adamMetricMetaUploaded = false;
		adamMetricScope = 0u;
	}

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
	if (d_adamRowSecond) { cudaFree(d_adamRowSecond); d_adamRowSecond = 0; }
	if (d_adamColSecond) { cudaFree(d_adamColSecond); d_adamColSecond = 0; }
	if (d_adamRowMetric) { cudaFree(d_adamRowMetric); d_adamRowMetric = 0; }
	if (d_adamColMetric) { cudaFree(d_adamColMetric); d_adamColMetric = 0; }
	if (d_adamRowStructMetric) { cudaFree(d_adamRowStructMetric); d_adamRowStructMetric = 0; }
	if (d_adamColStructMetric) { cudaFree(d_adamColStructMetric); d_adamColStructMetric = 0; }
	if (d_adamPrevMhat) { cudaFree(d_adamPrevMhat); d_adamPrevMhat = 0; }
	if (d_adamMetricScratch) { cudaFree(d_adamMetricScratch); d_adamMetricScratch = 0; }
	if (d_echoObserveEntries) { cudaFree(d_echoObserveEntries); d_echoObserveEntries = 0; }
	if (d_matraBatchItems) { cudaFree(d_matraBatchItems); d_matraBatchItems = 0; }
	if (d_matraStatsBatch) { cudaFree(d_matraStatsBatch); d_matraStatsBatch = 0; }
	if (d_matraCoreBatchPtrs) { cudaFree(d_matraCoreBatchPtrs); d_matraCoreBatchPtrs = 0; }
	if (d_matraStepBatchPtrs) { cudaFree(d_matraStepBatchPtrs); d_matraStepBatchPtrs = 0; }
	if (d_matraInfoBatch) { cudaFree(d_matraInfoBatch); d_matraInfoBatch = 0; }
	if (d_muonBatchItems) { cudaFree(d_muonBatchItems); d_muonBatchItems = 0; }
	if (d_muonCoreBatchPtrs) { cudaFree(d_muonCoreBatchPtrs); d_muonCoreBatchPtrs = 0; }
	if (d_muonStepBatchPtrs) { cudaFree(d_muonStepBatchPtrs); d_muonStepBatchPtrs = 0; }
	if (d_muonInfoBatch) { cudaFree(d_muonInfoBatch); d_muonInfoBatch = 0; }
	if (d_adamBaseLr) { cudaFree(d_adamBaseLr);  d_adamBaseLr = 0; }
	if (d_adamWd)     { cudaFree(d_adamWd);      d_adamWd     = 0; }
	if (d_adamGroupScales) { cudaFree(d_adamGroupScales); d_adamGroupScales = 0; }
	if (d_adamGroupPrevStepRms) { cudaFree(d_adamGroupPrevStepRms); d_adamGroupPrevStepRms = 0; }
	if (d_adamSizes)  { cudaFree(d_adamSizes);   d_adamSizes  = 0; }
	if (d_adamMetricRows) { cudaFree(d_adamMetricRows); d_adamMetricRows = 0; }
	if (d_adamMetricCols) { cudaFree(d_adamMetricCols); d_adamMetricCols = 0; }
	adamGroupCount = 0;
	adamMaxSize = 0;
	echoObserveCapacity = 0;
	matraCoreBatchCapacity = 0;
	muonCoreBatchCapacity = 0;
	echoObserveEntryCount = 0;
	echoObserveTotalFeatures = 0;
	echoObserveSeqLen = 0u;
	echoObserveScope = 0u;
	echoObserveTokenModel = false;
	echoObserveMetaUploaded = false;
	matraBatchDescriptorsUploaded = false;
	adamPtrsUploaded = false;
	adamMetricMetaUploaded = false;
	adamMetricScope = 0u;
	matraBatchDescriptorCount = 0;
	matraBatchDescriptorHash = 0ULL;
	initialized = false;
	lowpReady = false;
	lowpDType = 0;
	lowpIsCanonical = false;
	// GpuBuffer destructors handle cudaFree automatically.
}

// Populate every BF16 Lowp mirror from its FP32 master via the existing
// cast_f32_to_bf16 kernel. Allocates mirror buffers on first call.
bool GpuTransformerWeights::ensureLowpMirrors()
{
	if (!initialized)
		return false;

	// BF16-weights mode: *Lowp buffers ARE the canonical weight store.
	// Subsequent refreshes from FP32 master would overwrite the in-place
	// Adam updates.  First call (lowpReady=false) still goes through the
	// cast pass below to seed the mirrors from the just-uploaded FP32
	// init weights; later calls short-circuit.
	if (lowpIsCanonical && lowpReady) return true;

	// For each master -> mirror pair, allocate the mirror if empty and cast.
	// Helper captures the shape from the master buffer's allocated size.
#define GLADES_LOWP_ENSURE(master, mirror)                                 \
	do {                                                                   \
		const size_t n_ = (master).size();                                 \
		if (n_ == 0) { break; }                                            \
		if ((mirror).size() != n_) {                                       \
			if (!(mirror).allocate(n_)) return false;                      \
		}                                                                  \
		if (!cast_f32_to_bf16((master).data(), (mirror).data(), n_))       \
			return false;                                                  \
	} while (0)

	if (tokenModel)
		GLADES_LOWP_ENSURE(tokE, tokELowp);
	GLADES_LOWP_ENSURE(WIn, WInLowp);
	if (!tieEmbeddings)
		GLADES_LOWP_ENSURE(WOut, WOutLowp);

	for (unsigned int li = 0; li < nLayers; ++li)
	{
		Block& b = blocks[li];
		GLADES_LOWP_ENSURE(b.Wq, b.WqLowp);
		GLADES_LOWP_ENSURE(b.Wk, b.WkLowp);
		GLADES_LOWP_ENSURE(b.Wv, b.WvLowp);
		GLADES_LOWP_ENSURE(b.Wo, b.WoLowp);
		GLADES_LOWP_ENSURE(b.W1, b.W1Lowp);
		GLADES_LOWP_ENSURE(b.W2, b.W2Lowp);
	}
#undef GLADES_LOWP_ENSURE

	lowpReady = true;
	// lowpDType is set by the caller (sgd_transformer) based on
	// TrainingConfig.mixedPrecision.weightDType; only BF16 is supported here.
	return true;
}

// ---- GpuTransformerScratch ----

GpuTransformerScratch::GpuTransformerScratch()
    : initialized(false),
      T(0), dModel(0), dFF(0), dModelKV(0), nHeads(0), nLayers(0),
      inputSize(0), outSize(0), ff1Width(0),
      slotsPerLayer(0), nCheckpoints(0),
      d_dKdVZeroPtrs(0), d_dKdVZeroSizes(0)
{
}

GpuTransformerScratch::~GpuTransformerScratch()
{
	free();
}

bool ensureTransformerScratch(GpuTransformerScratch*& scratch,
                              const TransformerGpuScratchConfig& cfg)
{
	if (!scratch)
		scratch = new GpuTransformerScratch();
	if (!scratch)
		return false;

	// Effective slot count expected for this config (used for shape match).
	unsigned int expectedSlots = cfg.nLayers;
	if (cfg.activationCheckpoint && cfg.nLayers > 1u)
	{
		double k = std::ceil(std::sqrt(static_cast<double>(cfg.nLayers)));
		unsigned int K = static_cast<unsigned int>(k);
		if (K < 1u) K = 1u;
		if (K > cfg.nLayers) K = cfg.nLayers;
		expectedSlots = K;
	}
	const bool shapeMatches =
	    scratch->initialized &&
	    scratch->T >= cfg.T &&
	    scratch->inputSize == cfg.inputSize &&
	    scratch->outSize == cfg.outSize &&
	    scratch->dModel == cfg.dModel &&
	    scratch->dFF == cfg.dFF &&
	    scratch->dModelKV == cfg.dModelKV &&
	    scratch->nHeads == cfg.nHeads &&
	    scratch->nLayers == cfg.nLayers &&
	    scratch->ff1Width == cfg.ff1Width &&
	    scratch->slotsPerLayer == expectedSlots;
	if (shapeMatches)
		return true;

	return scratch->allocate(cfg.T, cfg.inputSize, cfg.outSize,
	                         cfg.dModel, cfg.dFF, cfg.dModelKV,
	                         cfg.nHeads, cfg.nLayers, cfg.ff1Width,
	                         cfg.activationCheckpoint);
}

bool GpuTransformerScratch::allocate(unsigned int newT, unsigned int is, unsigned int os,
                                      unsigned int dm, unsigned int df, unsigned int dmkv,
                                      unsigned int nh, unsigned int nl, unsigned int f1w,
                                      bool activationCheckpoint)
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

	// Activation gradient checkpointing: shrink per-layer activation scratches
	// to slotsPerLayer = ⌈√L⌉ slots (cyclic li%K addressing).  When disabled,
	// slotsPerLayer == nLayers — modulo collapses to identity, behavior
	// identical to pre-checkpoint allocation.
	if (activationCheckpoint && nl > 1u)
	{
		double k = std::ceil(std::sqrt(static_cast<double>(nl)));
		unsigned int K = static_cast<unsigned int>(k);
		if (K < 1u) K = 1u;
		if (K > nl) K = nl;
		slotsPerLayer = K;
		// Number of segments = ⌈nl / K⌉; checkpoints = nSegments - 1 (segment 0
		// reads from `h`, no checkpoint needed).
		const unsigned int nSegments = (nl + K - 1u) / K;
		nCheckpoints = (nSegments > 0u) ? (nSegments - 1u) : 0u;
	}
	else
	{
		slotsPerLayer = nl;
		nCheckpoints = 0u;
	}

	const size_t sT = static_cast<size_t>(T);
	const size_t sdm = static_cast<size_t>(dm);
	const size_t sdf = static_cast<size_t>(df);
	const size_t sdmkv = static_cast<size_t>(dmkv);
	const size_t snl = static_cast<size_t>(nl);
	const size_t sf1w = static_cast<size_t>(f1w);
	const size_t sis = static_cast<size_t>(is);
	const size_t sos = static_cast<size_t>(os);
	// Per-layer activation slot count (= snl when checkpointing off).
	const size_t sSlots = static_cast<size_t>(slotsPerLayer);
	const size_t sCkpt  = static_cast<size_t>(nCheckpoints);

	// Forward
	if (!x.allocate(sT * sis)) return false;
	if (!h.allocate(sT * sdm)) return false;
	if (!ln1Mean.allocate(sSlots * sT)) return false;
	if (!ln1InvStd.allocate(sSlots * sT)) return false;
	if (!x1.allocate(sSlots * sT * sdm)) return false;
	if (!Q.allocate(sSlots * sT * sdm)) return false;
	if (!K.allocate(sSlots * sT * sdmkv)) return false;
	if (!V.allocate(sSlots * sT * sdmkv)) return false;
	if (!attnConcat.allocate(sSlots * sT * sdm)) return false;
	if (!attnOut.allocate(sSlots * sT * sdm)) return false;
	if (!hAfterAttn.allocate(sSlots * sT * sdm)) return false;
	if (!ln2Mean.allocate(sSlots * sT)) return false;
	if (!ln2InvStd.allocate(sSlots * sT)) return false;
	if (!x2.allocate(sSlots * sT * sdm)) return false;
	if (!ff1.allocate(sSlots * sT * sf1w)) return false;
	if (!ff1Act.allocate(sSlots * sT * sdf)) return false;
	if (!ffOut.allocate(sSlots * sT * sdm)) return false;
	if (!hAfterFF.allocate(sSlots * sT * sdm)) return false;
	// Activation checkpoints: hAfterFF at every Kth segment boundary.
	if (sCkpt > 0u)
	{
		if (!checkpoints.allocate(sCkpt * sT * sdm)) return false;
	}
	(void)snl; // silence unused-when-checkpointing-on if compiler warns
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

	// BF16 activation staging, sized for the widest activation tile in the net
	// (T * max(dModel, ff1Width, vocabSize, inputSize)). Allocated
	// unconditionally so mpEnable can be toggled without re-allocating scratch.
	{
		size_t widest = sT * sdm;
		if (sT * sf1w > widest) widest = sT * sf1w;
		if (sT * sdf  > widest) widest = sT * sdf;
		if (sT * sos  > widest) widest = sT * sos;
		if (sT * sis  > widest) widest = sT * sis;
		if (!activationLowp.allocate(widest)) return false;
		if (!activationLowp2.allocate(widest)) return false;
		// Q/K/V BF16 scratches: one layer's worth each. Use max(dModel, dModelKV)
		// to cover both Q (dModel) and K/V (dModelKV) within a single dimension.
		const size_t qkvMax = (sdm > sdmkv ? sT * sdm : sT * sdmkv);
		if (!qLowp.allocate(sT * sdm)) return false;
		if (!kLowp.allocate(qkvMax)) return false;
		if (!vLowp.allocate(qkvMax)) return false;
	}

	// GPU loss computation scalars
	if (!lossSum.allocate(1)) return false;
	if (!lossCount.allocate(1)) return false;
	if (!correctCount.allocate(1)) return false;
	if (!validCount.allocate(1)) return false;
	if (!lossPack.allocate(4)) return false;

	// Shared FP32 grad-write scratch.  Sized to the widest weight tensor
	// across the model so any single backward GEMM can write here with
	// beta=0; bf16_accum_axpy then commits the result to the persistent
	// BF16 grad buffer (when MixedPrecisionConfig::gradStorageBf16=true).
	// Always allocated — it's ≤260 MB at 1.84B and simplifies the dispatch.
	{
		size_t widestWeight = sdm * sdm;                          // Wq, Wo
		if (sdm * sdmkv > widestWeight) widestWeight = sdm * sdmkv;  // Wk, Wv
		if (sf1w * sdm > widestWeight) widestWeight = sf1w * sdm;    // W1
		if (sdm * sdf  > widestWeight) widestWeight = sdm * sdf;     // W2
		if (sos * sdm > widestWeight) widestWeight = sos * sdm;      // tokE / WOut
		if (!gradScratchFp32.allocate(widestWeight)) return false;

		// Same shape; allocated only when weightStorageBf16 mode is on (caller
		// can re-allocate later via ensureWeightScratchFp32).  Initial alloc
		// here because it's tied to the model shape; caller toggles it on by
		// re-calling allocate() with the right config.
		// (Allocated unconditionally; 260 MB at 1.84B is small relative to
		// what we save by retiring FP32 weight masters, and the caller may
		// not have known weightStorageBf16 at scratch-allocate time.)
		if (!weightScratchFp32.allocate(widestWeight)) return false;
	}

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
                                    const float* b1, const float* b2,
                                    int mlaLatentDim,
                                    const float* Wdkv, const float* Wuk, const float* Wuv)
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

	// Paradigm #76 MLA: upload latent projections if active.
	if (mlaLatentDim > 0 && Wdkv && Wuk && Wuv) {
		const size_t dC = (size_t)mlaLatentDim;
		if (!b.Wdkv.upload(Wdkv, dm * dC)) return false;
		if (!b.Wuk.upload(Wuk,   dC * dmkv)) return false;
		if (!b.Wuv.upload(Wuv,   dC * dmkv)) return false;
	}

	return true;
}

bool uploadTransformerTokenIds(GpuTransformerScratch& scratch,
                               const int* tokenIds, size_t count)
{
	if (!scratch.initialized || !tokenIds)
		return false;
	return scratch.tokenIds.uploadAsync(tokenIds, count);
}

bool uploadTransformerDenseInputs(GpuTransformerScratch& scratch,
                                  const float* hostInputs, size_t count)
{
	if (!scratch.initialized || !hostInputs)
		return false;
	return scratch.x.uploadAsync(hostInputs, count);
}

bool uploadTransformerRopeInvFreq(GpuTransformerScratch& scratch,
                                  const float* invFreq, size_t count)
{
	if (!scratch.initialized || !invFreq)
		return false;
	return scratch.gpuInvFreq.uploadAsync(invFreq, count);
}

static bool download_host_buffer(const GpuBuffer<float>& src,
                                 const HostFloatBufferView& dst)
{
	if (dst.size == 0u)
		return true;
	if (!dst.data || !src.allocated() || src.size() < dst.size)
		return false;
	return src.download(dst.data, dst.size);
}

bool downloadTransformerWeightsToHost(const GpuTransformerWeights& gpu,
                                      const TransformerHostWeightsView& host)
{
	if (!downloadTransformerWeights(gpu,
	                                host.tokE.data, host.tokE.size,
	                                host.WIn.data, host.WIn.size,
	                                host.bIn.data, host.bIn.size,
	                                host.WOut.data, host.WOut.size,
	                                host.bOut.data, host.bOut.size,
	                                host.lmBias.data, host.lmBias.size,
	                                host.lnFinalGamma.data, host.lnFinalGamma.size,
	                                host.lnFinalBeta.data, host.lnFinalBeta.size))
	{
		return false;
	}

	if (host.blockCount != gpu.nLayers)
		return false;
	if (gpu.nLayers > 0u && !host.blocks)
		return false;

	for (unsigned int l = 0; l < gpu.nLayers; ++l)
	{
		const GpuTransformerWeights::Block& gb = gpu.blocks[l];
		const TransformerHostBlockWeightsView& hb = host.blocks[l];
		if (!download_host_buffer(gb.Wq, hb.Wq)) return false;
		if (!download_host_buffer(gb.Wk, hb.Wk)) return false;
		if (!download_host_buffer(gb.Wv, hb.Wv)) return false;
		if (!download_host_buffer(gb.Wo, hb.Wo)) return false;
		if (!download_host_buffer(gb.W1, hb.W1)) return false;
		if (!download_host_buffer(gb.W2, hb.W2)) return false;
		if (!download_host_buffer(gb.bq, hb.bq)) return false;
		if (!download_host_buffer(gb.bk, hb.bk)) return false;
		if (!download_host_buffer(gb.bv, hb.bv)) return false;
		if (!download_host_buffer(gb.bo, hb.bo)) return false;
		if (!download_host_buffer(gb.b1, hb.b1)) return false;
		if (!download_host_buffer(gb.b2, hb.b2)) return false;
		if (!download_host_buffer(gb.ln1Gamma, hb.ln1Gamma)) return false;
		if (!download_host_buffer(gb.ln1Beta, hb.ln1Beta)) return false;
		if (!download_host_buffer(gb.ln2Gamma, hb.ln2Gamma)) return false;
		if (!download_host_buffer(gb.ln2Beta, hb.ln2Beta)) return false;
	}

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

bool zeroTransformerGradientsBf16(GpuTransformerWeights& gpu)
{
	if (!gpu.initialized) return false;
	// Phase-2: zero only the BF16 grad mirrors that exist (allocated when
	// MixedPrecisionConfig::gradStorageBf16=true).  bf16 element is 2 bytes;
	// cudaMemset of 0 yields exact bf16 zero (sign=0, exp=0, mant=0).
	if (gpu.gTokE_bf16.allocated())
		cudaMemset(gpu.gTokE_bf16.data(), 0,
		    gpu.gTokE_bf16.size() * sizeof(uint16_t));
	if (gpu.gWIn_bf16.allocated())
		cudaMemset(gpu.gWIn_bf16.data(), 0,
		    gpu.gWIn_bf16.size() * sizeof(uint16_t));
	if (gpu.gWOut_bf16.allocated())
		cudaMemset(gpu.gWOut_bf16.data(), 0,
		    gpu.gWOut_bf16.size() * sizeof(uint16_t));
	for (unsigned int l = 0; l < gpu.nLayers; ++l)
	{
		GpuTransformerWeights::Block& b = gpu.blocks[l];
		if (b.gWq_bf16.allocated())
			cudaMemset(b.gWq_bf16.data(), 0,
			    b.gWq_bf16.size() * sizeof(uint16_t));
		if (b.gWk_bf16.allocated())
			cudaMemset(b.gWk_bf16.data(), 0,
			    b.gWk_bf16.size() * sizeof(uint16_t));
		if (b.gWv_bf16.allocated())
			cudaMemset(b.gWv_bf16.data(), 0,
			    b.gWv_bf16.size() * sizeof(uint16_t));
		if (b.gWo_bf16.allocated())
			cudaMemset(b.gWo_bf16.data(), 0,
			    b.gWo_bf16.size() * sizeof(uint16_t));
		if (b.gW1_bf16.allocated())
			cudaMemset(b.gW1_bf16.data(), 0,
			    b.gW1_bf16.size() * sizeof(uint16_t));
		if (b.gW2_bf16.allocated())
			cudaMemset(b.gW2_bf16.data(), 0,
			    b.gW2_bf16.size() * sizeof(uint16_t));
		if (b.gWdkv_bf16.allocated())
			cudaMemset(b.gWdkv_bf16.data(), 0,
			    b.gWdkv_bf16.size() * sizeof(uint16_t));
		if (b.gWuk_bf16.allocated())
			cudaMemset(b.gWuk_bf16.data(), 0,
			    b.gWuk_bf16.size() * sizeof(uint16_t));
		if (b.gWuv_bf16.allocated())
			cudaMemset(b.gWuv_bf16.data(), 0,
			    b.gWuv_bf16.size() * sizeof(uint16_t));
	}
	return true;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
