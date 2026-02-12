// GPU mirror of TensorRNNState / TensorGatedState (Phase 5).
#pragma once

#include "gpu_buffer.h"

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

struct GpuRNNWeights
{
	bool initialized;
	unsigned int numHidden;
	unsigned int inputSize;
	unsigned int outSize;

	struct Hidden
	{
		unsigned int in;
		unsigned int h;
		GpuBuffer<float> Wxh;   // [h, in]
		GpuBuffer<float> Whh;   // [h, h]
		GpuBuffer<float> vWxh;
		GpuBuffer<float> vWhh;
		GpuBuffer<float> gWxh;
		GpuBuffer<float> gWhh;
		GpuBuffer<float> bias;  // [h]
		GpuBuffer<float> gBias;
	};

	struct Out
	{
		unsigned int in;
		unsigned int out;
		GpuBuffer<float> Why;   // [out, in]
		GpuBuffer<float> vWhy;
		GpuBuffer<float> gWhy;
		GpuBuffer<float> bias;  // [out]
		GpuBuffer<float> gBias;
	};

	Hidden* hiddenLayers;
	Out outputLayer;

	GpuRNNWeights() : initialized(false), numHidden(0), inputSize(0), outSize(0), hiddenLayers(0) {}
	~GpuRNNWeights() { free(); }

	bool allocate(unsigned int inputSize, unsigned int outSize,
	              const unsigned int* hiddenSizes, unsigned int numHidden);
	void free();
};

// Gated variant (GRU: gateCount=3, LSTM: gateCount=4).
struct GpuGatedWeights
{
	bool initialized;
	unsigned int numHidden;
	unsigned int inputSize;
	unsigned int outSize;
	unsigned int gateCount;

	struct Hidden
	{
		unsigned int in;
		unsigned int h;
		GpuBuffer<float> W;    // [gateCount, h, in]
		GpuBuffer<float> U;    // [gateCount, h, h]
		GpuBuffer<float> vW;
		GpuBuffer<float> vU;
		GpuBuffer<float> gW;
		GpuBuffer<float> gU;
		GpuBuffer<float> bias; // [gateCount, h]
		GpuBuffer<float> gBias;
	};

	struct Out
	{
		unsigned int in;
		unsigned int out;
		GpuBuffer<float> Why;   // [out, in]
		GpuBuffer<float> vWhy;
		GpuBuffer<float> gWhy;
		GpuBuffer<float> bias;  // [out]
		GpuBuffer<float> gBias;
	};

	Hidden* hiddenLayers;
	Out outputLayer;

	GpuGatedWeights() : initialized(false), numHidden(0), inputSize(0), outSize(0), gateCount(0), hiddenLayers(0) {}
	~GpuGatedWeights() { free(); }

	bool allocate(unsigned int inputSize, unsigned int outSize,
	              const unsigned int* hiddenSizes, unsigned int numHidden,
	              unsigned int gateCount);
	void free();
};

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

struct GpuRNNWeights
{
	bool initialized;
	GpuRNNWeights() : initialized(false) {}
};

struct GpuGatedWeights
{
	bool initialized;
	GpuGatedWeights() : initialized(false) {}
};

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
