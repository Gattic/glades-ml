// GPU DFF state implementation.
#include "gpu_dff_state.h"

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

bool GpuDFFWeights::allocate(const unsigned int* layerSizes, unsigned int numLayers)
{
	free();
	if (numLayers < 2)
		return false;

	numTransitions = numLayers - 1;
	transitions = new Transition[numTransitions];

	for (unsigned int t = 0; t < numTransitions; ++t)
	{
		Transition& tr = transitions[t];
		tr.in = layerSizes[t];
		tr.out = layerSizes[t + 1];
		const size_t wSize = static_cast<size_t>(tr.out) * static_cast<size_t>(tr.in);

		if (!tr.W.allocate(wSize)) return false;
		if (!tr.vW.allocate(wSize)) return false;
		if (!tr.gW.allocate(wSize)) return false;
		if (!tr.bias.allocate(tr.out)) return false;
		if (!tr.gBias.allocate(tr.out)) return false;
	}

	initialized = true;
	return true;
}

void GpuDFFWeights::free()
{
	if (transitions)
	{
		delete[] transitions;
		transitions = 0;
	}
	numTransitions = 0;
	initialized = false;
}

bool GpuDFFScratch::allocate(const unsigned int* layerSizes, unsigned int nl)
{
	free();
	if (nl == 0)
		return false;

	numLayers = nl;
	a = new GpuBuffer<float>[nl];
	delta = new GpuBuffer<float>[nl];

	for (unsigned int i = 0; i < nl; ++i)
	{
		if (!a[i].allocate(layerSizes[i])) return false;
		if (!delta[i].allocate(layerSizes[i])) return false;
	}

	initialized = true;
	return true;
}

void GpuDFFScratch::free()
{
	if (a)
	{
		delete[] a;
		a = 0;
	}
	if (delta)
	{
		delete[] delta;
		delta = 0;
	}
	numLayers = 0;
	initialized = false;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
