// GPU mirror of TensorCNNState.
#pragma once

#include "gpu_buffer.h"

#include <memory>

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

struct GpuCNNWeights
{
	bool initialized;
	unsigned int numConvLayers;
	unsigned int numFCLayers;

	struct ConvLayer
	{
		unsigned int outC, inC, kH, kW;
		GpuBuffer<float> W;       // [outC, inC*kH*kW]
		GpuBuffer<float> bias;    // [outC]
		GpuBuffer<float> gW;
		GpuBuffer<float> gBias;
		GpuBuffer<float> vW;
		GpuBuffer<float> v2W;
		GpuBuffer<float> vBias;
		GpuBuffer<float> v2Bias;
		// BatchNorm
		GpuBuffer<float> bnGamma;
		GpuBuffer<float> bnBeta;
		GpuBuffer<float> bnRunMean;
		GpuBuffer<float> bnRunVar;
		GpuBuffer<float> gBnGamma;
		GpuBuffer<float> gBnBeta;
		GpuBuffer<float> vBnGamma;
		GpuBuffer<float> v2BnGamma;
		GpuBuffer<float> vBnBeta;
		GpuBuffer<float> v2BnBeta;
	};

	struct FCLayer
	{
		unsigned int in, out;
		GpuBuffer<float> W;       // [out, in]
		GpuBuffer<float> bias;    // [out]
		GpuBuffer<float> gW;
		GpuBuffer<float> gBias;
		GpuBuffer<float> vW;
		GpuBuffer<float> v2W;
		GpuBuffer<float> vBias;
		GpuBuffer<float> v2Bias;
	};

	std::unique_ptr<ConvLayer[]> convLayers;
	std::unique_ptr<FCLayer[]> fcLayers;

	GpuCNNWeights() : initialized(false), numConvLayers(0), numFCLayers(0) {}
	~GpuCNNWeights() { free(); }

	void free()
	{
		convLayers.reset();
		fcLayers.reset();
		numConvLayers = 0;
		numFCLayers = 0;
		initialized = false;
	}
};

struct GpuCNNScratch
{
	bool initialized;

	struct ConvLayerScratch
	{
		GpuBuffer<float> im2col;
		GpuBuffer<float> convOut;
		GpuBuffer<float> bnOut;
		GpuBuffer<float> bnMean;
		GpuBuffer<float> bnInvStd;
		GpuBuffer<float> bnNorm;
		GpuBuffer<float> actOut;
		GpuBuffer<float> poolOut;
		GpuBuffer<int> poolArgmax;
		GpuBuffer<float> dPoolOut;
		GpuBuffer<float> dActOut;
		GpuBuffer<float> dConvOut;
		GpuBuffer<float> dIm2col;
	};

	unsigned int numConvLayers;
	std::unique_ptr<ConvLayerScratch[]> convScratch;

	GpuCNNScratch() : initialized(false), numConvLayers(0) {}
	~GpuCNNScratch() { free(); }

	void free()
	{
		convScratch.reset();
		numConvLayers = 0;
		initialized = false;
	}
};

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

struct GpuCNNWeights
{
	bool initialized;
	GpuCNNWeights() : initialized(false) {}
};

struct GpuCNNScratch
{
	bool initialized;
	GpuCNNScratch() : initialized(false) {}
};

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
