// Internal transformer training helpers shared by sgd_transformer.cpp.
#ifndef _GLADES_TRANSFORMER_TRAIN_DETAIL_H
#define _GLADES_TRANSFORMER_TRAIN_DETAIL_H

#include <stdint.h>
#include <vector>

namespace glades {

struct LearningRateScheduleConfig;

namespace transformer_train_detail {

struct DoubleBufferView
{
	const double* data;
	unsigned int size;

	DoubleBufferView()
	    : data(NULL), size(0u)
	{
	}
};

struct LinearWeightView
{
	const float* weights;
	unsigned int weightCount;
	const uint16_t* lowpWeights;
	unsigned int lowpWeightCount;
	const float* bias;
	unsigned int biasCount;
	bool useLowpWeights;
	int lowpDType;

	LinearWeightView()
	    : weights(NULL),
	      weightCount(0u),
	      lowpWeights(NULL),
	      lowpWeightCount(0u),
	      bias(NULL),
	      biasCount(0u),
	      useLowpWeights(false),
	      lowpDType(0)
	{
	}
};

float clip_maybe(float v, float limit);
float transformer_schedule_multiplier(const glades::LearningRateScheduleConfig& schedule,
                                      int epochIdx,
                                      unsigned int stepInEpoch,
                                      unsigned int totalStepsInEpoch);

void add_positional_encoding(float* h,
                             unsigned int T,
                             unsigned int dModel,
                             const DoubleBufferView& invDenomPair);

void linear_forward_maybe_lowp(const float* X,
                               unsigned int T,
                               unsigned int inSize,
                               const LinearWeightView& weights,
                               unsigned int outSize,
                               float* Y);

void linear_backward_accum_maybe_lowp(const float* X,
                                      const float* dY,
                                      unsigned int T,
                                      unsigned int inSize,
                                      unsigned int outSize,
                                      std::vector<float>& gW,
                                      std::vector<float>& gB,
                                      const LinearWeightView& weights,
                                      float* dXOut);

struct AttnFwdCtx
{
	const float* Q;
	const float* K;
	const float* V;
	float* O;
	unsigned int dModel;
	unsigned int dModelKV;
	unsigned int dHead;
	unsigned int nHeads;
	unsigned int nKVHeads;
	unsigned int T;
	unsigned int groupSize;
	bool causal;
	const unsigned char* keyAllowed;
	// Paradigm shift #78 ATTENTION-SINK + sliding window. Both default to 0 (disabled).
	unsigned int sinkCount;
	unsigned int windowSize;
};

void attn_fwd_body(void* ud, unsigned int begin, unsigned int end);

struct AttnBwdCtx
{
	const float* Q;
	const float* K;
	const float* V;
	const float* dO;
	float* dQ;
	float* dK;
	float* dV;
	unsigned int dModel;
	unsigned int dModelKV;
	unsigned int dHead;
	unsigned int nHeads;
	unsigned int nKVHeads;
	unsigned int T;
	unsigned int groupSize;
	bool causal;
	const unsigned char* keyAllowed;
	unsigned int nChunksPerHead;
	unsigned int totalItems;
	float* dKVscratch;
	// Paradigm shift #78 ATTENTION-SINK + sliding window. Both default to 0 (disabled).
	unsigned int sinkCount;
	unsigned int windowSize;
};

void attn_bwd_body(void* ud, unsigned int begin, unsigned int end);

struct AttnBwdReduceCtx
{
	const float* dKVscratch;
	float* dK;
	float* dV;
	unsigned int dHead;
	unsigned int dModelKV;
	unsigned int T;
	unsigned int nHeads;
	unsigned int nKVHeads;
	unsigned int groupSize;
	unsigned int nChunksPerHead;
};

void attn_bwd_reduce_body(void* ud, unsigned int begin, unsigned int end);

struct RopeFwdCtx
{
	float* buf;
	unsigned int T;
	unsigned int rowStride;
	unsigned int dHead;
	unsigned int ropeDim;
	DoubleBufferView invFreq;
	bool inverse;
};

void rope_body(void* ud, unsigned int begin, unsigned int end);

struct NormFwdCtx
{
	const float* X;
	unsigned int D;
	const float* gamma;
	const float* beta;
	unsigned int gammaSize;
	unsigned int betaSize;
	float eps;
	float* Y;
	float* meanOut;
	float* invStdOut;
	bool isRmsNorm;
};

void norm_fwd_body(void* ud, unsigned int begin, unsigned int end);

struct TiedEmbLogitsCtx
{
	const float* H;
	unsigned int dModel;
	const float* tokE;
	const float* lmBias;
	unsigned int lmBiasSize;
	unsigned int vocab;
	float* logitsOut;
};

void tied_emb_logits_body(void* ud, unsigned int begin, unsigned int end);

} // namespace transformer_train_detail
} // namespace glades

#endif
