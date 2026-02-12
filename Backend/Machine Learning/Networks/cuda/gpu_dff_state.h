// GPU mirror of TensorDFFState (Phase 4).
#pragma once

#include "gpu_buffer.h"

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

struct GpuDFFWeights
{
	bool initialized;
	unsigned int numTransitions;

	struct Transition
	{
		unsigned int in;
		unsigned int out;
		GpuBuffer<float> W;     // [out, in]
		GpuBuffer<float> vW;
		GpuBuffer<float> gW;
		GpuBuffer<float> bias;  // [out]
		GpuBuffer<float> gBias;
	};

	Transition* transitions;

	GpuDFFWeights() : initialized(false), numTransitions(0), transitions(0) {}
	~GpuDFFWeights() { free(); }

	bool allocate(const unsigned int* layerSizes, unsigned int numLayers);
	void free();
};

struct GpuDFFScratch
{
	bool initialized;
	unsigned int numLayers;

	// Per-layer activations and deltas.
	GpuBuffer<float>* a;     // a[li] has layerSizes[li] elements
	GpuBuffer<float>* delta; // delta[li] for non-input layers

	GpuDFFScratch() : initialized(false), numLayers(0), a(0), delta(0) {}
	~GpuDFFScratch() { free(); }

	bool allocate(const unsigned int* layerSizes, unsigned int numLayers);
	void free();
};

} // namespace gpu
} // namespace glades

#else // !GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

struct GpuDFFWeights
{
	bool initialized;
	GpuDFFWeights() : initialized(false) {}
};

struct GpuDFFScratch
{
	bool initialized;
	GpuDFFScratch() : initialized(false) {}
};

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
