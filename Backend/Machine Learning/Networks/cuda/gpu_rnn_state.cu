// GPU RNN/Gated state implementation.
#include "gpu_rnn_state.h"

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

// ---- GpuRNNWeights ----

bool GpuRNNWeights::allocate(unsigned int is, unsigned int os,
                               const unsigned int* hiddenSizes, unsigned int nh)
{
	free();
	inputSize = is;
	outSize = os;
	numHidden = nh;

	hiddenLayers = new Hidden[nh];
	for (unsigned int i = 0; i < nh; ++i)
	{
		Hidden& hl = hiddenLayers[i];
		hl.in = (i == 0) ? is : hiddenSizes[i - 1];
		hl.h = hiddenSizes[i];
		const size_t wxhSize = static_cast<size_t>(hl.h) * static_cast<size_t>(hl.in);
		const size_t whhSize = static_cast<size_t>(hl.h) * static_cast<size_t>(hl.h);

		if (!hl.Wxh.allocate(wxhSize)) return false;
		if (!hl.Whh.allocate(whhSize)) return false;
		if (!hl.vWxh.allocate(wxhSize)) return false;
		if (!hl.vWhh.allocate(whhSize)) return false;
		if (!hl.gWxh.allocate(wxhSize)) return false;
		if (!hl.gWhh.allocate(whhSize)) return false;
		if (!hl.bias.allocate(hl.h)) return false;
		if (!hl.gBias.allocate(hl.h)) return false;
	}

	// Output layer
	outputLayer.in = (nh > 0) ? hiddenSizes[nh - 1] : is;
	outputLayer.out = os;
	const size_t whySize = static_cast<size_t>(os) * static_cast<size_t>(outputLayer.in);
	if (!outputLayer.Why.allocate(whySize)) return false;
	if (!outputLayer.vWhy.allocate(whySize)) return false;
	if (!outputLayer.gWhy.allocate(whySize)) return false;
	if (!outputLayer.bias.allocate(os)) return false;
	if (!outputLayer.gBias.allocate(os)) return false;

	initialized = true;
	return true;
}

void GpuRNNWeights::free()
{
	if (hiddenLayers)
	{
		delete[] hiddenLayers;
		hiddenLayers = 0;
	}
	numHidden = 0;
	initialized = false;
}

// ---- GpuGatedWeights ----

bool GpuGatedWeights::allocate(unsigned int is, unsigned int os,
                                 const unsigned int* hiddenSizes, unsigned int nh,
                                 unsigned int gc)
{
	free();
	inputSize = is;
	outSize = os;
	numHidden = nh;
	gateCount = gc;

	hiddenLayers = new Hidden[nh];
	for (unsigned int i = 0; i < nh; ++i)
	{
		Hidden& hl = hiddenLayers[i];
		hl.in = (i == 0) ? is : hiddenSizes[i - 1];
		hl.h = hiddenSizes[i];
		const size_t wSize = static_cast<size_t>(gc) * static_cast<size_t>(hl.h) * static_cast<size_t>(hl.in);
		const size_t uSize = static_cast<size_t>(gc) * static_cast<size_t>(hl.h) * static_cast<size_t>(hl.h);
		const size_t biasSize = static_cast<size_t>(gc) * static_cast<size_t>(hl.h);

		if (!hl.W.allocate(wSize)) return false;
		if (!hl.U.allocate(uSize)) return false;
		if (!hl.vW.allocate(wSize)) return false;
		if (!hl.vU.allocate(uSize)) return false;
		if (!hl.gW.allocate(wSize)) return false;
		if (!hl.gU.allocate(uSize)) return false;
		if (!hl.bias.allocate(biasSize)) return false;
		if (!hl.gBias.allocate(biasSize)) return false;
	}

	// Output layer
	outputLayer.in = (nh > 0) ? hiddenSizes[nh - 1] : is;
	outputLayer.out = os;
	const size_t whySize = static_cast<size_t>(os) * static_cast<size_t>(outputLayer.in);
	if (!outputLayer.Why.allocate(whySize)) return false;
	if (!outputLayer.vWhy.allocate(whySize)) return false;
	if (!outputLayer.gWhy.allocate(whySize)) return false;
	if (!outputLayer.bias.allocate(os)) return false;
	if (!outputLayer.gBias.allocate(os)) return false;

	initialized = true;
	return true;
}

void GpuGatedWeights::free()
{
	if (hiddenLayers)
	{
		delete[] hiddenLayers;
		hiddenLayers = 0;
	}
	numHidden = 0;
	initialized = false;
}

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
