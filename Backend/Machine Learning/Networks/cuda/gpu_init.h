// GPU-side parameter initialization (curand-backed).
//
// Replaces the host-side InitGlorot / glades::rng::normal loops in
// NNetwork::ensureTensorParametersInitialized for the --gpu path.
//
// Determinism:
//   Each tensor takes a 64-bit `tensor_id` mixed with the network seed via a
//   golden-ratio constant. Per-thread Philox state is keyed on
//   (seed XOR tensor_id*phi, tid).  Given fixed launch geometry (grid/block),
//   bit-exact output is reproducible across runs for a given (seed, tensor_id).
//
//   Determinism is preserved across NVIDIA GPU generations as long as
//   curandStatePhilox4_32_10_t is used (counter-based, layout-independent).
//   Layout-dependent across grid/block changes — these init helpers fix
//   geometry to (kBlocks, kBlock) below.
#pragma once

#include <cstddef>
#include <stdint.h>

#ifdef GLADES_HAVE_CUDA

namespace glades {
namespace gpu {

// Glorot/Xavier uniform: W ~ U(-limit, +limit), limit = sqrt(6/(fanIn+fanOut)).
// Matches the host `InitGlorot::run` distribution (uniform, symmetric).
void initGlorotUniform(float* d_W, std::size_t N,
                       unsigned int fanIn, unsigned int fanOut,
                       uint64_t seed, uint64_t tensor_id);

// Truncated-normal-style normal init (no truncation): W ~ N(mean, stddev).
// Used for token embedding init (mean=0, stddev=0.02 standard LLM practice).
void initNormal(float* d_W, std::size_t N, float mean, float stddev,
                uint64_t seed, uint64_t tensor_id);

// Zero-fill (Adam state, biases, gradients).  Equivalent to cudaMemsetAsync.
void initZeros(float* d_W, std::size_t N);

// Constant-fill (LayerNorm gamma=1, etc.).  Falls through to memset for val==0.
void initConstant(float* d_W, std::size_t N, float val);

} // namespace gpu
} // namespace glades

#endif // GLADES_HAVE_CUDA
